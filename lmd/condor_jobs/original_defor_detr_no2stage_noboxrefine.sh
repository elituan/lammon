#!/bin/bash

cd /home/ehoang/eli/LMMCT/lmd/
export PYTHONPATH=.

MY_SUPER_FLAGS="mode=FAST_RUN,device=gpu,floatX=float32"
/home/ehoang/miniconda3/envs/lmmct/bin/python /home/ehoang/eli/LMMCT/lmd/main.py --devices 4 --batch_size 5