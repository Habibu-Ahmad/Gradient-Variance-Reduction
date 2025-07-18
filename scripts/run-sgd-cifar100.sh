#!/bin/bash
# Train SGD on CIFAR-100 with ResNet-18

python train.py \
    --arch resnet18 \
    --datasets CIFAR100 \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.05 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --cutout \
    --alpha 0.01 \
    --optimizer sgd \
    --seed 42 \
    --gpu 0 \
    --log-file sgd_resnet18_cifar100.log \
    --data-root ./datasets/
