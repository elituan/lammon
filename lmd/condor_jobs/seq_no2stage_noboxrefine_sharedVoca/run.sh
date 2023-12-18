#!/bin/bash

cd /home/ehoang/eli/LMMCT/lmd/
export PYTHONPATH=.

MY_SUPER_FLAGS="mode=FAST_RUN,device=gpu,floatX=float32"
/home/ehoang/miniconda3/envs/lmmct/bin/python /home/ehoang/eli/LMMCT/lmd/main.py --devices 3 --batch_size 5  --cls_loss_coef 7 --bbox_loss_coef 0 --giou_loss_coef 0 --check_val_every_n_epoch 1