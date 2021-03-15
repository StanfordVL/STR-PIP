#!/bin/bash

gpu_id=0
split='train'
n_acts=1

n_epochs=80
start_epoch=2
lr=1e-4
wd=1e-5
bt=1
lr_decay=1
decay_every=20
n_workers=0
save_every=10
evaluate_every=1

seq_len=30
predict=1
pred_seq_len=30
predict_k=0

annot_loc_format='/sailhome/agalczak/crossing/datasets/annot_{}_loc.pkl'

load_cache='masks'
cache_format='/sailhome/agalczak/crossing/datasets/cache/jaad_loc/{}/ped{}_fid{}.pkl'
save_cache_format=$cache_format

pretrained_path='/sailhome/agalczak/crossing/ckpts/JAAD_loc/loc_concat_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt1_posNone_branchboth_collapse0_combinepair_tmp/best_pred.pth'


use_gru=1
use_trn=0
ped_gru=1
# pos_mode='center'
pos_mode='none'
use_act=0
use_gt_act=0
use_pose=0
branch='both'
collapse_cls=0
combine_method='pair'
suffix='_tmp'
ckpt_name='branch'$branch'_collapse'$collapse_cls'_combine'$combine_method$suffix

CUDA_VISIBLE_DEVICES=$gpu_id python3 -m train.py \
  --model='loc_concat' \
  --split=$split \
  --n-acts=$n_acts \
  --device=$gpu_id \
  --dset-name='JAAD_loc' \
  --ckpt-name=$ckpt_name \
  --n-epochs=$n_epochs \
  --start-epoch=$start_epoch \
  --lr-init=$lr \
  --wd=$wd \
  --lr-decay=$lr_decay \
  --decay-every=$decay_every \
  --seq-len=$seq_len \
  --predict=$predict \
  --pred-seq-len=$pred_seq_len \
  --predict-k=$predict_k \
  --batch-size=$bt \
  --save-every=$save_every \
  --evaluate-every=$evaluate_every \
  --n-workers=$n_workers \
  --annot-loc-format=$annot_loc_format \
  --load-cache=$load_cache \
  --cache-format=$cache_format \
  --save-cache-format=$save_cache_format \
  --branch=$branch \
  --collapse-cls=$collapse_cls \
  --combine-method=$combine_method \
  --use-gru=$use_gru \
  --use-trn=$use_trn \
  --ped-gru=$ped_gru \
  --pos-mode=$pos_mode \
  --use-act=$use_act \
  --use-gt-act=$use_gt_act \
  --use-pose=$use_pose \
