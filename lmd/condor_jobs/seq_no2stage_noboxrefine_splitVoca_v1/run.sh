#!/bin/bash

cd /home/ehoang/eli/LMMCT/lmd/
export PYTHONPATH=.

MY_SUPER_FLAGS="mode=FAST_RUN,device=gpu,floatX=float32"
/home/ehoang/miniconda3/envs/lmmct/bin/python /home/ehoang/eli/LMMCT/lmd/main.py --devices_ 5 --batch_size 4 \
--cls_loss_coef 2 --bbox_loss_coef 5 --giou_loss_coef 2 --check_val_every_n_epoch 1  --focal_gamma 2 \
--focal_alpha 0.25 --log_every_n_steps 50 --accumulate_batches 0_1 --cost_feats_weight 2_2_2_2_1 --lr 2e-4  \
--max_epochs 80 --lr_decay_steps 100 --lr_decay_rate 1 --swa 5 --load_v_num 178