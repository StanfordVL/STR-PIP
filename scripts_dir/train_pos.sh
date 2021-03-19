#!/bin/bash

gpu_id=0
split='train'
n_acts=9

lr=1e-4
wd=1e-5
bt=16
lr_decay=1
decay_every=20
n_workers=0
save_every=10
evaluate_every=1

seq_len=30
predict=1
pred_seq_len=30

load_cache='pos'
# masks
cache_format='/sailhome/bingbin/STR-PIP/datasets/cache/jaad_collapse_max/{}/ped{}_fid{}.pkl'

use_gru=1
branch='both'
pos_mode='none'
use_gt_act=1
collapse_cls=0
combine_method='pair'
suffix='_'$pos_mode'_9acts_withGTAct_noPos'
ckpt_name='branch'$branch'_collapse'$collapse_cls'_combine'$combine_method$suffix

CUDA_VISIBLE_DEVICES=$gpu_id python3 -m train.py \
  --model='pos' \
  --split=$split \
  --n-acts=$n_acts \
  --device=$gpu_id \
  --dset-name='JAAD' \
  --ckpt-name=$ckpt_name \
  --lr-init=$lr \
  --wd=$wd \
  --lr-decay=$lr_decay \
  --decay-every=$decay_every \
  --seq-len=$seq_len \
  --predict=$predict \
  --pred-seq-len=$pred_seq_len \
  --batch-size=$bt \
  --save-every=$save_every \
  --evaluate-every=$evaluate_every \
  --n-workers=$n_workers \
  --load-cache=$load_cache \
  --cache-format=$cache_format \
  --branch=$branch \
  --pos-mode=$pos_mode \
  --use-gt-act=$use_gt_act \
  --collapse-cls=$collapse_cls \
  --combine-method=$combine_method \
  --use-gru=$use_gru \
