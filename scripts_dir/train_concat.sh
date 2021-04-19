#!/bin/bash
#!/usr/bin/env python3

gpu_id=0
split='train'
n_acts=9

lr=1e-4
wd=1e-5
bt=4
lr_decay=1
decay_every=20
n_workers=0
save_every=10
evaluate_every=1

seq_len=30
predict=1
pred_seq_len=30
predict_k=0

annot_ped_format='/sailhome/ajarno/STR-PIP/datasets/annot_{}_ped_withTag_sanityNoPose.pkl'

load_cache='masks'
cache_format='/sailhome/ajarno/STR-PIP/datasets/cache/jaad_collapse_max/{}/ped{}_fid{}.pkl'
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
suffix='_cacheMasks_fixGRU_eval3_9acts_noAct_sanityWithPose_withReLU_pedGRU'
suffix='_test_tmp'
ckpt_name='branch'$branch'_collapse'$collapse_cls'_combine'$combine_method$suffix

WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$gpu_id python3 -m train.py \
  --model='concat' \
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
  --predict-k=$predict_k \
  --batch-size=$bt \
  --save-every=$save_every \
  --evaluate-every=$evaluate_every \
  --n-workers=$n_workers \
  --annot-ped-format=$annot_ped_format \
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
