#!/bin/bash

mode='evaluate'
# mode='extract'

gpu_id=0
n_workers=0
n_acts=9

seq_len=30
predict=1
pred_seq_len=30

# test only
slide=0
rand_test=1
log_every=10
ckpt_dir='/sailhome/bingbin/STR-PIP/ckpts'
dataset='JAAD'

split='test'
use_gru=1
ped_gru=1
use_trn=0
pos_mode='none'
use_act=0
use_gt_act=0
use_pose=0
# branch='ped'
branch='both'
collapse_cls=0
combine_method='pair'

load_cache='masks'
# load_cache='none'

cache_format='/sailhome/bingbin/STR-PIP/datasets/cache/jaad_collapse/{}/ped{}_fid{}.pkl'
save_cache_format=$cache_format

# cache_format='/sailhome/bingbin/STR-PIP/datasets/cache/jaad_collapse_max/{}/ped{}_fid{}.pkl'
# cache_format='/sailhome/bingbin/STR-PIP/datasets/cache/jaad_collapse/{}/ped{}_fid{}.pkl'

# ckpt_name='concat_gru_lr1.0e-04_wd1.0e-05_bt4_ped_collapse0_combinepair_useBBox1_cacheMasks_fixGRU'
# which_epoch=13

# ckpt_name='concat_gru_lr1.0e-04_wd1.0e-05_bt4_ped_collapse0_combinepair_useBBox0_cacheMasks_fixGRU'
# which_epoch=44

# ckpt_name='concat_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt4_posNone_branchped_collapse0_combinepair_cacheMasks_fixGRU_eval3_9acts_withGTAct'

# bkwfairi
# ckpt_name='concat_gru_seq14_pred1_lr1.0e-04_wd1.0e-05_bt4_posNone_branchped_collapse0_combinepair_cacheMasks_fixGRU_eval3_9acts_noAct_sanityWithPose_withReLU'

# dur1n8v7
# saved: pred ~74.9
# ckpt_name='concat_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt4_posNone_branchboth_collapse0_combinepair_cacheMasks_fixGRU_eval3_9acts_noAct_sanityWithPose_withReLU_pedGRU'

# -1 for the best epoch
which_epoch=-1

# this is to set a non-existent epoch s.t. the features are extracted from ImageNet backbone
# which_epoch=100

if [ $which_epoch -eq -1 ]
then
  epoch_name='best_pred'
else
  epoch_name=$which_epoch
fi
save_output=10
save_output_format=$ckpt_dir'/'$dataset'/'$ckpt_name'/output_epoch'$epoch_name'_step{}.pkl'


if [ "$mode" = "extract" ]
then
  extract_feats_dir='/sailhome/bingbin/STR-PIP/datasets/cache/JAAD_conv_feats/'$ckpt_name'/'$split'/'
else
  extract_feats_dir='none_existent'
fi

CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py \
  --model='concat' \
  --split=$split \
  --n-acts=$n_acts \
  --mode=$mode \
  --device=$gpu_id \
  --log-every=$log_every \
  --dset-name='JAAD' \
  --ckpt-name=$ckpt_name \
  --batch-size=1 \
  --n-workers=$n_workers \
  --load-cache=$load_cache \
  --save-cache-format=$save_cache_format \
  --cache-format=$cache_format \
  --seq-len=$seq_len \
  --predict=$predict \
  --pred-seq-len=$pred_seq_len \
  --use-gru=$use_gru \
  --ped-gru=$ped_gru \
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
  --save-output=$save_output \
  --save-output-format=$save_output_format \
  --extract-feats-dir=$extract_feats_dir

