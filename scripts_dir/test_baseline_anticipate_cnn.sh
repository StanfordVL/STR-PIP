#!/bin/bash

mode='evaluate'

gpu_id=1
split='test'
n_acts=9
n_workers=0
bt=1

# test only
slide=0
rand_test=1
log_every=10
ckpt_dir='/sailhome/bingbin/crossing/ckpts'
dataset='JAAD'

ckpt_name="baseline_anticipate_cnn_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt4_posNone_branchped_collapse0_combinepair_noGTAct_addDetLoss"

seq_len=30
predict=1
pred_seq_len=30

load_cache='masks'
# masks
cache_format='/sailhome/bingbin/crossing/datasets/cache/jaad_collapse_max/{}/ped{}_fid{}.pkl'

which_epoch=-1
if [ $which_epoch -eq -1 ]
then
  epoch_name='best_pred'
else
  epoch_name=$which_epoch
fi
save_output=1
save_output_format=$ckpt_dir'/'$dataset'/'$ckpt_name'/output_epoch'$epoch_name'_step{}.pkl'

if [ "$mode" = "extract" ]
then
  extract_feats_dir='/sailhome/bingbin/crossing/datasets/cache/JAAD_conv_feats/concat_gru_lr1.0e-05_bt4_test_epoch5/test/'
else
  extract_feats_dir='none_existent'
fi


use_gru=1
use_gt_act=0
branch='ped'
pos_mode='none'
collapse_cls=0
combine_method='pair'

CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py \
  --model='baseline_anticipate_cnn' \
  --mode=$mode \
  --slide=$slide \
  --rand-test=$rand_test \
  --split=$split \
  --n-acts=$n_acts \
  --device=$gpu_id \
  --dset-name='JAAD' \
  --ckpt-dir=$ckpt_dir \
  --ckpt-name=$ckpt_name \
  --which-epoch=$which_epoch \
  --save-output=$save_output \
  --save-output-format=$save_output_format \
  --seq-len=$seq_len \
  --predict=$predict \
  --pred-seq-len=$pred_seq_len \
  --batch-size=$bt \
  --n-workers=$n_workers \
  --load-cache=$load_cache \
  --cache-format=$cache_format \
  --branch=$branch \
  --pos-mode=$pos_mode \
  --collapse-cls=$collapse_cls \
  --combine-method=$combine_method \
  --use-gru=$use_gru \
  --use-gt-act=$use_gt_act \
