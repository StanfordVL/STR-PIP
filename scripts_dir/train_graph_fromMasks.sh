#!/bin/bash

gpu_id=0
n_workers=0
split='train'
n_acts=9

# train only
n_epochs=80
lr=1e-4
wd=1e-5
bt=6
lr_decay=1
decay_every=30
save_every=10
evaluate_every=1
log_every=30

load_cache='masks'
# load_cache='feats'

if [ $load_cache = 'feats' ]
then
  cache_root='/sailhome/bingbin/STR-PIP/datasets/cache/JAAD_conv_feats/'
  cache_baseformat='/{}/ped{}_fid{}.pkl'
  # v3 feats
  cache_dir='concat_gru_seq14_pred1_lr1.0e-04_wd1.0e-05_bt4_posNone_branchped_collapse0_combinepair_cacheMasks_fixGRU_eval3_9acts_noAct_sanityWithPose_withReLU_ordered'

  # v2 feats
  # cache_format='concat_gru_lr1.0e-04_wd1.0e-05_bt4_ped_collapse0_combinepair_useBBox0_cacheMasks_fixGRU_singleTime'

  # cache_format='imageNet_pretrained_singleTime'
  cache_format=$cache_root$cache_dir$cache_baseformat
else
  cache_format='/sailhome/bingbin/STR-PIP/datasets/cache/jaad_collapse/{}/ped{}_fid{}.pkl'
fi


# Regularization on temporal smoothness
reg_smooth='none'
reg_lambda=5e-4

seq_len=30
predict=1
pred_seq_len=30
predict_k=0

# temporal modeling
use_gru=1
use_trn=0
ped_gru=1
ctxt_gru=0

# features
use_act=0
use_gt_act=0
pos_mode='none'
branch='ped'
adj_type='spatial'
#adj_type='inner'
n_layers=2
collapse_cls=0
combine_method='pair'

# saving & loading
suffix='_v3Feats_fromMasks_pedGRU'
# suffix='_v3Feats__pedGRU_ctxtGRU'
ckpt_name='branch'$branch'_collapse'$collapse_cls'_combine'$combine_method'_adjType'$adj_type'_nLayers'$n_layers$suffix
# ckpt_name='graph_seq30_layer2_embed'

CUDA_VISIBLE_DEVICES=$gpu_id python3 -m train.py \
  --model='graph' \
  --reg-smooth=$reg_smooth \
  --reg-lambda=$reg_lambda \
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
  --predict-k=$predict_k \
  --batch-size=$bt \
  --save-every=$save_every \
  --evaluate-every=$evaluate_every \
  --log-every=$log_every \
  --n-workers=$n_workers \
  --load-cache=$load_cache \
  --cache-format=$cache_format \
  --branch=$branch \
  --adj-type=$adj_type \
  --n-layers=$n_layers \
  --collapse-cls=$collapse_cls \
  --combine-method=$combine_method \
  --use-gru=$use_gru \
  --use-trn=$use_trn \
  --ped-gru=$ped_gru \
  --ctxt-gru=$ctxt_gru \
  --use-act=$use_act \
  --use-gt-act=$use_gt_act \
  --pos-mode=$pos_mode
