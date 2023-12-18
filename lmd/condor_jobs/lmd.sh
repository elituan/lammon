#!/bin/bash

#cd ~/home/ehoang/eli/LMMCT/lmd/
export PYTHONPATH=.

MY_SUPER_FLAGS="mode=FAST_RUN,device=gpu,floatX=float32"
/home/ehoang/miniconda3/envs/lmmct/bin/python /home/ehoang/eli/LMMCT/lmd/main.py --fast_dev_run --devices_ 4