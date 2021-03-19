#!/bin/bash

gpu_id=0
split='train'
n_acts=1

n_epochs=100
lr=1e-4
wd=1e-7
bt=8
lr_decay=1
decay_every=20
n_workers=0
save_every=10
evaluate_every=1
fc_in_dim=256

seq_len=14
predict=1
pred_seq_len=1

load_cache='pos'
# masks
cache_format='/sailhome/bingbin/STR-PIP/datasets/cache/jaad_collapse_max/{}/ped{}_fid{}.pkl'
save_cache_format=$cache_format

use_gru=1
use_gt_act=0
use_pose=1
branch='ped'
pos_mode='none'
collapse_cls=0
combine_method='pair'
suffix='_gru'$fc_in_dim'_zeroPad'
ckpt_name='branch'$branch'_collapse'$collapse_cls'_combine'$combine_method$suffix

CUDA_VISIBLE_DEVICES=$gpu_id python3 -m train.py \
  --model='baseline_pose' \
  --split=$split \
  --n-acts=$n_acts \
  --device=$gpu_id \
  --dset-name='JAAD' \
  --ckpt-name=$ckpt_name \
  --n-epochs=$n_epochs \
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
  --fc-in-dim=$fc_in_dim \
  --n-workers=$n_workers \
  --load-cache=$load_cache \
  --cache-format=$cache_format \
  --save-cache-format=$save_cache_format \
  --branch=$branch \
  --pos-mode=$pos_mode \
  --collapse-cls=$collapse_cls \
  --combine-method=$combine_method \
  --use-gru=$use_gru \
  --use-gt-act=$use_gt_act \
  --use-pose=$use_pose \
