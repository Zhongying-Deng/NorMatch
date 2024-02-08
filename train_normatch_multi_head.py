import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
from models.flow_model import FlowGMM


logger = logging.getLogger(__name__)
best_acc = 0

class FlowGMM2(FlowGMM):
    def forward(self, x):
        z1 = self.net(x)
        z_all = z1.reshape((len(z1), -1))
        return  self.loss_fn.prior.class_logits(z_all)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn', 'stl10', 
                                'imagenet', 'mini_imagenet'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext', 'resnet'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lambda-flow', default=1, type=float,
                        help='coefficient of flow GMM loss')
    parser.add_argument('--lambda-flow-unsup', default=0., type=float,
                        help='coefficient of unsupervised log likelyhood loss for Flow GMM')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--mixing', action='store_true',
                        help="use channel shuffle and mixing or not")
    parser.add_argument('--flow-dist-trainable', action='store_true',
                        help="train parameters (mean/std/weights) of GMM distribution in NFlow")
    parser.add_argument('--dist_align', action='store_true',
                        help="use distribution alignment or not")
    parser.add_argument('--warmup_flow', default=5, type=float,
                        help='warmup epochs for FlowGMM')
    parser.add_argument('--use_two_flows', action='store_true',
                        help='use two FlowGMM classifiers or not')
    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        else:
            import models.resnet as models
            # resnet18 for STL10
            if args.dataset == 'stl10' or args.dataset == 'mini_imagenet':
                model = models.resnet18(num_classes=args.num_classes, pretrained=False)
            else:
                model = models.resnet50(num_classes=args.num_classes, pretrained=False)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device if torch.cuda.is_available() else torch.device('cpu')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10' or args.dataset == 'svhn':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
    elif args.dataset == 'stl10':
        args.num_classes = 10
    elif args.dataset == 'mini_imagenet':
        args.num_classes = 100
    else:
        args.num_classes = 1000

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if args.dataset == 'stl10':
        from dataset.stl10 import STL10_GETTERS
        labeled_dataset, unlabeled_dataset, test_dataset = STL10_GETTERS[args.dataset](
            args, './data/stl10')
    elif args.dataset == 'imagenet':
        from dataset.imagenet import ImageNet_GETTERS
        labeled_dataset, unlabeled_dataset, test_dataset = ImageNet_GETTERS[args.dataset](
            args, './data/imagenet')
    elif args.dataset == 'mini_imagenet':
        from dataset.mini_imagenet import MiniImageNet_GETTERS
        labeled_dataset, unlabeled_dataset, test_dataset = MiniImageNet_GETTERS[args.dataset](
            args, './data/imagenet/mini_imagenet')
    else:
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
            args, './data') 

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)
    if args.arch == 'wideresnet':
        dim = model.channels
    elif args.arch == 'resnext':
        dim = model.stages[3]
    else:
        dim = model.fdim

    mean, inv_cov_stds, weights = None, None, None
    mean2, inv_cov_stds2, weights2 = None, None, None
    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        if args.flow_dist_trainable:
            mean = checkpoint['flow_mean']
            inv_cov_stds = checkpoint['flow_std']
            weights = checkpoint['flow_weights']
            if args.use_two_flows:
                mean2 = checkpoint['flow_mean2']
                inv_cov_stds2 = checkpoint['flow_std2']
                weights2 = checkpoint['flow_weights2']
    flow_model = FlowGMM(dim, args.num_classes, args, 
        mean, inv_cov_stds, weights)
        
    logger.info("Total params of FlowGMM: {:.2f}M".format(
            sum(p.numel() for p in flow_model.parameters())/1e6))
    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)
    flow_model.to(args.device)

    flow_model.prior.means.requires_grad = args.flow_dist_trainable
    flow_model.prior.weights.requires_grad = args.flow_dist_trainable
    flow_model.prior.inv_cov_stds.requires_grad = args.flow_dist_trainable

    if args.use_two_flows:
        flow_model2 = FlowGMM(dim, args.num_classes, args, 
                              mean2, inv_cov_stds2, weights2)
        flow_model2.to(args.device)
        flow_model2.prior.means.requires_grad = args.flow_dist_trainable
        flow_model2.prior.weights.requires_grad = args.flow_dist_trainable
        flow_model2.prior.inv_cov_stds.requires_grad = args.flow_dist_trainable
    else:
        flow_model2 = None

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0},
        #{'params': [p for n, p in flow_model.named_parameters()], 
        #    'lr': 0.1 * args.lr, 'weight_decay': args.wdecay}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    param_flow = [{'params': [p for n, p in flow_model.named_parameters()],
             'weight_decay': args.wdecay},
                  {'params':  [flow_model.prior.inv_cov_stds, flow_model.prior.means, flow_model.prior.weights],
                      'weight_decay': args.wdecay}
    ]
    if args.use_two_flows:
        param_flow += [{'params': [p for n, p in flow_model2.named_parameters()],
             'weight_decay': args.wdecay},
                  {'params':  [flow_model2.prior.inv_cov_stds, flow_model2.prior.means, flow_model2.prior.weights],
                      'weight_decay': args.wdecay}
        ]
    

    optimizer_flow = optim.AdamW(param_flow, lr=0.001)  # lr=0.0005

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)
    scheduler_flow = get_cosine_schedule_with_warmup(
        optimizer_flow, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        print('Resume epoch: {}'.format(args.start_epoch))
        model.load_state_dict(checkpoint['state_dict'])
        flow_model.load_state_dict(checkpoint['flow_state_dict'])
        if args.use_two_flows:
            flow_model2.load_state_dict(checkpoint['flow2_state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_flow.load_state_dict(checkpoint['flow_optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scheduler_flow.load_state_dict(checkpoint['flow_scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        flow_model = torch.nn.parallel.DistributedDataParallel(
            flow_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        if args.use_two_flows:
            flow_model2 = torch.nn.parallel.DistributedDataParallel(
                flow_model2, device_ids=[args.local_rank],
                output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, flow_model, optimizer, optimizer_flow, ema_model, scheduler, scheduler_flow, flow_model2)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, flow_model, optimizer, optimizer_flow, ema_model, scheduler, 
          scheduler_flow, flow_model2=None):
    if args.amp:
        from apex import amp
    global best_acc
    num_high_conf = 0.
    num_total = 0.
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    flow_model.train()
    if flow_model2 is not None:
        flow_model2.train()
    if args.mixing:
        mixing = True
    else:
        mixing = None
    prob_list = []
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_flow = AverageMeter()
        losses_unsup = AverageMeter()
        mask_probs = AverageMeter()
        lambda_flow = min(epoch / 5., 1.) * args.lambda_flow
        lambda_flow_unsup = min(epoch / 5., 1.) * args.lambda_flow_unsup
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            feats, logits = model(inputs, mixing=mixing, return_feat=True)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits
            feats = de_interleave(feats, 2*args.mu+1)
            feats_x = feats[:batch_size]
            feats_u_w, feats_u_s = feats[batch_size:].chunk(2)
            loss_flow, _ = flow_model(feats_x.detach(), targets_x, return_unsup_loss=True)
            #loss_unsup = flow_model(feats_u_w.detach())

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            if args.dist_align:
                prob_list.append(pseudo_label.mean(0))
                if len(prob_list)>32:
                    prob_list.pop(0)
                prob_avg = torch.stack(prob_list,dim=0).mean(0)
                pseudo_label = pseudo_label / prob_avg
                pseudo_label = pseudo_label / pseudo_label.sum(dim=1, keepdim=True)

            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask_thresh = max_probs.ge(args.threshold).float()
            num_high_conf += mask_thresh.sum()
            num_total += max_probs.size(0)
            loss_unsup, logits_u_flow = flow_model(feats_u_w.detach())
            logits_u_flow = F.softmax(logits_u_flow, 1)
            probs_flow, targets_u_flow = torch.max(logits_u_flow, dim=-1)
            
            if args.use_two_flows:
                loss_flow2, _ = flow_model2(feats_x.detach(), targets_x, return_unsup_loss=True)
                loss_flow = (loss_flow + loss_flow2) / 2.
                _, logits_u_sampled = flow_model2(feats_u_w.detach())
            else:
                logits_u_sampled = flow_model.sample_classifier(feats_u_w.detach())
            logits_u_sampled = F.softmax(logits_u_sampled, 1)
            probs_sampled, targets_u_sampled = torch.max(logits_u_sampled, dim=-1)

            mask = (targets_u_flow == targets_u).float()
            mask *= (targets_u_sampled == targets_u).float()
            min_probs = torch.min(probs_sampled, probs_flow)
            tmp = (1-mask) * torch.min(max_probs, min_probs)
            mask = torch.max(mask, tmp)

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()
            loss = Lx + args.lambda_u * Lu + lambda_flow * loss_flow +  lambda_flow_unsup * loss_unsup

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_flow.update(loss_flow.item())
            losses_unsup.update(loss_unsup.item())
            optimizer.step()
            scheduler.step()
            clip_grad_norm_(flow_model.parameters(), max_norm=50, norm_type=2)
            if flow_model2 is not None:
                clip_grad_norm_(flow_model2.parameters(), max_norm=50, norm_type=2)
            optimizer_flow.step()
            scheduler_flow.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            flow_model.zero_grad()
            if flow_model2 is not None:
                flow_model2.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Loss_f: {loss_f:.4f} Mask: {mask:.2f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_f=losses_flow.avg,
                    mask=mask_probs.avg))
                p_bar.update()
            else:
                if batch_idx % 100 == 0:
                    print("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Loss_f: {loss_f:.4f}, Loss_unsup: {loss_unsup:.4f}, Mask: {mask:.2f}.".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        loss_f=losses_flow.avg,
                        loss_unsup=losses_unsup.avg,
                        mask=mask_probs.avg)
                        )
        print('High Confidence Ratio: {ratio:.2f}'.format(
            ratio=100*(num_high_conf/num_total)
            ))
        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, flow_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            flow_model_to_save = flow_model.module if hasattr(flow_model, "module") else flow_model
            if args.use_two_flows:
                flow_model2_to_save = flow_model2.module if hasattr(flow_model2, "module") else flow_model2
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'flow_state_dict': flow_model_to_save.state_dict(),
                'flow_mean': flow_model_to_save.prior.means,
                'flow_std': flow_model_to_save.prior.inv_cov_stds,
                'flow_weights': flow_model_to_save.prior.weights,
                'flow2_state_dict': flow_model2_to_save.state_dict() if args.use_two_flows else None,
                'flow_mean2': flow_model2_to_save.prior.means if args.use_two_flows else None,
                'flow_std2': flow_model2_to_save.prior.inv_cov_stds if args.use_two_flows else None,
                'flow_weights2': flow_model2_to_save.prior.weights if args.use_two_flows else None,
                'flow_optimizer': optimizer_flow.state_dict(),
                'flow_scheduler': scheduler_flow.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, flow_model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_flow = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        flow_model.eval()
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            feats, outputs = model(inputs, return_feat=True)
            outputs_flow = flow_model.predict(feats)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            prec1, prec5 = accuracy(outputs_flow, targets, topk=(1, 5))
            top1_flow.update(prec1.item(), inputs.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    logger.info("top-1 flow model acc: {:.2f}".format(top1_flow.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()

