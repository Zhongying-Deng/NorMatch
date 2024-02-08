option=0
if (( $option == 0 )); then
python train_normatch.py --dataset cifar10 --num-labeled 40 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 2 \
 --out ./result_ssl_cifar/cifar10_nflow@40_da_ema_onehot  --no-progress --flow-dist-trainable --lambda-flow-unsup 0.000001 \
 --dist_align  --no_onehot  --use-ema \
 2>&1| tee -a ./result_ssl_cifar/cifar10_nflow@40_da_ema_onehot_seed2.log &
python train_normatch.py --dataset cifar10 --num-labeled 250 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 2 \
 --out ./result_ssl_cifar/cifar10_nflow@250_da_ema_onehot  --no-progress --flow-dist-trainable --lambda-flow-unsup 0.000001 \
 --dist_align  --no_onehot --use-ema \
 2>&1| tee -a ./result_ssl_cifar/cifar10_nflow@250_da_ema_onehot_seed2.log &
python train_normatch.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 2 \
 --out ./result_ssl_cifar/cifar10_nflow@4000_da_ema_onehot  --no-progress --flow-dist-trainable --lambda-flow-unsup 0.000001 \
 --dist_align  --no_onehot  --use-ema \
 2>&1| tee -a ./result_ssl_cifar/cifar10_nflow@4000_da_ema_onehot_seed2.log &
fi
if (( $option == 1 )); then
python train_normatch.py --dataset cifar100 --num-labeled 10000 --arch wideresnet --batch-size 64 --lr 0.03 --wdecay 0.001 --expand-labels --seed 1 --out ./result_ssl_cifar/cifar100@10000_da_ema_onehot_seed1 --no-progress --flow-dist-trainable --lambda-flow-unsup 0.000001 --dist_align --use-ema --no_onehot   2>&1| tee -a ./result_ssl_cifar/cifar100@10000_da_ema_onehot_seed1.log &
python train_normatch.py --dataset cifar100 --num-labeled 2500 --arch wideresnet --batch-size 64 --lr 0.03 --wdecay 0.001 --expand-labels --seed 1 --out ./result_ssl_cifar/cifar100@2500_da_ema_onehot_seed1 --no-progress --flow-dist-trainable --lambda-flow-unsup 0.000001 --dist_align --use-ema --no_onehot   2>&1| tee -a ./result_ssl_cifar/cifar100@2500_da_ema_onehot_seed1.log &
python train_normatch.py --dataset cifar100 --num-labeled 400 --arch wideresnet --batch-size 64 --lr 0.03 --wdecay 0.001 --expand-labels --seed 1 --out ./result_ssl_cifar/cifar100@400_da_ema_onehot_seed1 --no-progress --flow-dist-trainable --lambda-flow-unsup 0.000001 --dist_align --use-ema --no_onehot  2>&1| tee -a ./result_ssl_cifar/cifar100@400_da_ema_onehot_seed1.log &
fi
if (( $option == 2 )); then
python train_normatch.py --dataset cifar100 --num-labeled 10000 --arch wideresnet --batch-size 64 --lr 0.03 --wdecay 0.001 --expand-labels --seed 3 --out ./result_ssl_cifar/cifar100@10000_da_ema_onehot_seed3 --no-progress --flow-dist-trainable --lambda-flow-unsup 0.000001 --dist_align --use-ema --no_onehot   2>&1| tee -a ./result_ssl_cifar/cifar100@10000_da_ema_onehot_seed3.log &
python train_normatch.py --dataset cifar100 --num-labeled 2500 --arch wideresnet --batch-size 64 --lr 0.03 --wdecay 0.001 --expand-labels --seed 3 --out ./result_ssl_cifar/cifar100@2500_da_ema_onehot_seed3 --no-progress --flow-dist-trainable --lambda-flow-unsup 0.000001 --dist_align --use-ema --no_onehot  2>&1| tee -a ./result_ssl_cifar/cifar100@2500_da_ema_onehot_seed3.log &
python train_normatch.py --dataset cifar100 --num-labeled 400 --arch wideresnet --batch-size 64 --lr 0.03 --wdecay 0.001 --expand-labels --seed 3 --out ./result_ssl_cifar/cifar100@400_da_ema_onehot_seed3 --no-progress --flow-dist-trainable --lambda-flow-unsup 0.000001 --dist_align --use-ema --no_onehot  2>&1| tee -a ./result_ssl_cifar/cifar100@400_da_ema_onehot_seed3.log &
fi
