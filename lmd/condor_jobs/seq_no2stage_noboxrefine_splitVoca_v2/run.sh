#!/bin/bash

cd /home/ehoang/eli/LMMCT/lmd/
export PYTHONPATH=.

MY_SUPER_FLAGS="mode=FAST_RUN,device=gpu,floatX=float32"
/home/ehoang/miniconda3/envs/lmmct/bin/python /home/ehoang/eli/LMMCT/lmd/main.py --devices 4 --batch_size 5 \
--cls_loss_coef 5 --bbox_loss_coef 2 --giou_loss_coef 2 --check_val_every_n_epoch 1  --focal_gamma 2 \
--focal_alpha 0.25 --log_every_n_steps 50 --accumulate_batches 0_2 --cost_feats_weight 1_1_1_1_1 --lr 2.2e-04  \
--max_epochs 15