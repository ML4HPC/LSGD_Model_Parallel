#!/bin/sh

#srun python _LSGD.py -a resnet50 --epoch 100 --batch-size 64 --gpu-num 8 --lr 6.4 /global/cscratch1/sd/kwangmin/dataset/ImageNet/ILSVRC2012


srun python m_LSGD.py -a resnet50 --epoch 90 --batch-size 32 --train-workers 7 --lr 0.1 /global/cscratch1/sd/kwangmin/dataset/ImageNet/ILSVRC2012 
