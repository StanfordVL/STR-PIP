#!/bin/bash

gpu_id=0
split='train'
n_acts=1

lr=1e-5
wd=1e-5
bt=2
lr_decay=1
decay_every=10
n_workers=0
save_every=10
evaluate_every=1

seq_len=30
predict=1
pred_seq_len=10
predict_k=0

annot_ped_format='/vision/group/prolix/processed/pedestrians/{}.pkl'

cache_obj_bbox_format='/vision/group/prolix/processed/obj_bbox_20fps_merged/{}_seg{}.pkl'
cache_obj_bbox_format_left='/vision/group/prolix/processed/left/obj_bbox_20fps_merged/{}_seg{}.pkl'
cache_obj_bbox_format_right='/vision/group/prolix/processed/right/obj_bbox_20fps_merged/{}_seg{}.pkl'

load_cache='mask'
cache_format=/dev/null
save_cache_format=$cache_format

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
suffix='_decay10_tmp'
ckpt_name='branch'$branch'_collapse'$collapse_cls'_combine'$combine_method$suffix

WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$gpu_id python -m train.py \
  --model='concat' \
  --split=$split \
  --n-acts=$n_acts \
  --device=$gpu_id \
  --dset-name='STIP' \
  --ckpt-name=$ckpt_name \
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
  --annot-ped-format=$annot_ped_format \
  --load-cache=$load_cache \
  --cache-obj-bbox-format=$cache_obj_bbox_format \
  --cache-obj-bbox-format-left=$cache_obj_bbox_format_left \
  --cache-obj-bbox-format-right=$cache_obj_bbox_format_right \
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
