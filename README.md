# NorMatch
Official PyTorch implementation of [NorMatch: Matching Normalizing Flows with Discriminative Classifiers for Semi-Supervised Learning](https://openreview.net/forum?id=ebiAFpQ0Lw&noteId=5PmBQKApbT) which has been accepted to Transactions on Machine Learning Research (TMLR).

The code is based on 1) [Semi-Supervised Learning with Normalizing Flows](https://invertibleworkshop.github.io/accepted_papers/pdfs/INNF_2019_paper_28.pdf) of which the implementation is [here](https://github.com/izmailovpavel/flowgmm.git); 2) [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685) and its [pytorch implementation](https://github.com/kekmodel/FixMatch-pytorch).

## Training

### Requirements
- python 3.6+ (python 3.10.13 is actually used)
- torch 1.12.1
- torchvision 0.13.1
- cudatoolkit 11.3.1
- tensorboard
- numpy
- tqdm
- scipy
- torchcontrib
- apex (optional)

To install the above packages, please run the following instructions.
```bash
conda create -n normatch python=3.10.13
conda activate normatch
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
conda install tensorboard tqdm scipy
pip install torchcontrib
```

### Data preparation

Create a folder named `data`. STL10 and CIFAR-10/100 can be automatically downloaded when training model. Mini-ImageNet should be prepared according to [Label Propagation for Deep Semi-Supervised Learning](https://github.com/ahmetius/LP-DeepSSL). The folder structure looks like 
```
|-- data
    |-- stl10
    |-- imagenet
        |-- mini_imagenet
    |-- cifar-100-python
    |-- cifar-10-batches-py
```

### Training models

The training script is `train_normatch.sh`. A simple example in the script to train the model by 40 labeled data of CIFAR-10 dataset is:

```python
python train_normatch.py --dataset cifar10 --num-labeled 40 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 2 \
 --out ./result_ssl_cifar/cifar10_nflow@40_da_ema_onehot  --no-progress --lambda-flow-unsup 0.000001 \
 --flow-dist-trainable --use-ema --dist_align  --no_onehot
```
The path specified by `--out` will be created to save the checkpoints. `--no_onehot` applies to the datasets excluding the STL10 (see `Implementation Details` of Section 4.1 in our paper: "on STL-10 where a one-hot version is used"). `--dist_align` is not used in the ablation study of our paper.

To resume from a checkpoint, please add the `--resume <path_to_ckpt>` option.

To use one more FlowGMM, please use the `train_normatch_multi_head.py` file together with `--use_two_flows` option.

To train a FixMatch baseline, please use `train_fixmatch.sh`. 


## Citations
```
@article{deng2022normatch,
  title={NorMatch: Matching Normalizing Flows with Discriminative Classifiers for Semi-Supervised Learning},
  author={Deng, Zhongying and Ke, Rihuan and Schonlieb, Carola-Bibiane and Aviles-Rivero, Angelica I},
  journal={arXiv preprint arXiv:2211.09593},
  year={2022}
}
```
