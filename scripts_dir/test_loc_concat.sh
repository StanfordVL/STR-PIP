#!/bin/bash

# mode='evaluate'
mode='extract'

gpu_id=0
n_workers=0
n_acts=1

seq_len=1
predict=0
pred_seq_len=30

# test only
slide=0
rand_test=1
log_every=10


split='train'
# split='test'
use_gru=1
use_trn=0
pos_mode='none'
use_act=0
use_gt_act=0
use_pose=0
# branch='ped'
branch='both'
collapse_cls=0
combine_method='pair'

annot_loc_format='/sailhome/bingbin/STR-PIP/datasets/annot_{}_loc.pkl'

load_cache='masks'
# load_cache='none'

cache_format='/sailhome/bingbin/STR-PIP/datasets/cache/jaad_loc/{}/ped{}_fid{}.pkl'
save_cache_format=$cache_format

ckpt_name='loc_concat_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt1_posNone_branchboth_collapse0_combinepair_tmp'

# -1 for the best epoch
which_epoch=-1

# this is to set a non-existent epoch s.t. the features are extracted from ImageNet backbone
# which_epoch=100

if [ "$mode" = "extract" ]
then
  extract_feats_dir='/sailhome/bingbin/STR-PIP/datasets/cache/jaad_loc/JAAD_conv_feats/'$ckpt_name'/'$split'/'
else
  extract_feats_dir='none_existent'
fi

CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py \
  --model='loc_concat' \
  --split=$split \
  --n-acts=$n_acts \
  --mode=$mode \
  --device=$gpu_id \
  --log-every=$log_every \
  --dset-name='JAAD_loc' \
  --ckpt-name=$ckpt_name \
  --batch-size=1 \
  --n-workers=$n_workers \
  --annot-loc-format=$annot_loc_format \
  --load-cache=$load_cache \
  --save-cache-format=$save_cache_format \
  --cache-format=$cache_format \
  --seq-len=$seq_len \
  --predict=$predict \
  --pred-seq-len=$pred_seq_len \
  --use-gru=$use_gru \
  --use-trn=$use_trn \
  --use-act=$use_act \
  --use-gt-act=$use_gt_act \
  --use-pose=$use_pose \
  --pos-mode=$pos_mode \
  --collapse-cls=$collapse_cls \
  --slide=$slide \
  --rand-test=$rand_test \
  --branch=$branch \
  --which-epoch=$which_epoch \
  --extract-feats-dir=$extract_feats_dir

