#!/bin/bash

cd /home/ehoang/eli/LMMCT/lmd/
export PYTHONPATH=.

MY_SUPER_FLAGS="mode=FAST_RUN,device=gpu,floatX=float32"
/home/ehoang/miniconda3/envs/lmmct/bin/python /home/ehoang/eli/LMMCT/lmd/main.py --devices 4 --batch_size 2 \
--cls_loss_coef 7 --bbox_loss_coef 0 --giou_loss_coef 0 --check_val_every_n_epoch 1 --no_shared_voca --focal_gamma 2 \
--focal_alpha 0.25 --log_every_n_steps 50 --accumulate_batches 0_1 --cost_feats_weight 2_2_2_2_1 --lr 1.1e-04  \
--max_epochs 10 --load_v_num 133