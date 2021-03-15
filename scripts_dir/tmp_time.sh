#!/bin/bash

gpu_id=0

lr=1e-5
wd=1e-5
bt=4
n_workers=0
save_every=1

use_gru=1

branch='ctxt'
ckpt_name=$branch

CUDA_VISIBLE_DEVICES=$gpu_id python3 -m tmp_time.py \
  --model='concat' \
  --device=$gpu_id \
  --dset-name='JAAD' \
  --ckpt-name=$ckpt_name \
  --lr-init=$lr \
  --wd=$wd \
  --batch-size=$bt \
  --save-every=$save_every \
  --n-workers=$n_workers \
  --branch=$branch \
  --use-gru=$use_gru
