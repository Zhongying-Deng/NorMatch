import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_ssl import RealNVP, RealNVPTabular, SSLGaussMixture, FlowLoss


class FlowGMM(nn.Module):
    def __init__(self, dim, n_cls, args, means=None, inv_cov_stds=None, weights=None):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda', args.gpu_id)
        else:
            self.device = torch.device('cpu')
        if means is not None:
            self.means = means
        else:
            self.means = torch.zeros((n_cls, dim)).to(self.device)
            for i in range(n_cls):
                self.means[i] = torch.randn(dim).to(self.device)
        self.net = RealNVPTabular(num_coupling_layers=6,in_dim=dim,hidden_dim=32,num_layers=1,dropout=True)
        self.prior = SSLGaussMixture(self.means, inv_cov_stds=inv_cov_stds, weights=weights )
        self.loss_fn = FlowLoss(self.prior)

    def forward(self, x, label=None, mask=None, return_unsup_loss=False):
        z1 = self.net(x)
        try:
            sldj = self.net.module.logdet()
        except:
            sldj = self.net.logdet()
        loss_unsup = self.loss_fn(z1, sldj=sldj)
        z_all = z1.reshape((len(z1), -1))
        if label is None:
            return loss_unsup, self.loss_fn.prior.class_logits(z_all)
        if mask is None:
            mask = torch.ones_like(label)
        labeled_mask = (mask != 0)
        if sum(labeled_mask) > 0:
            z_labeled = z_all[labeled_mask]
            y_labeled = label[labeled_mask]
            logits_all = self.loss_fn.prior.class_logits(z_all)
            logits_labeled = logits_all[labeled_mask]
            loss_nll = F.cross_entropy(logits_labeled, y_labeled)
            if not torch.isfinite(loss_nll).all():
               import pdb
               pdb.set_trace()
        else:
            loss_nll = torch.tensor(0.).to(self.device)
        if return_unsup_loss:
            return loss_nll, loss_unsup
        return loss_nll

    def predict(self, x):
        z1 = self.net(x)
        z_all = z1.reshape((len(z1), -1))
        pred = self.loss_fn.prior.class_logits(z_all)
        return pred
    
    def sample(self, batch_size, cls=None):
        with torch.no_grad():
            if cls is not None:
                z = self.prior.sample((batch_size,), gaussian_id=cls)
            else:
                z = self.prior.sample((batch_size,))
            try:
                x = self.net.module.inverse(z)
            except:
                x = self.net.inverse(z)

        return x

    def sample_classifier(self, x):
        z1 = self.net(x)
        z_all = z1.reshape((len(z1), -1))
        fc_sampled = self.loss_fn.prior.sample_classifier()
        pred = fc_sampled(z_all)
        return pred

