#!/bin/bash

gpu_id=1
split='train'
n_workers=0
n_acts=1
bt=8
log_every=30

# Training only
n_epochs=80
start_epoch=0
lr=1e-4
wd=1e-5
lr_decay=1
decay_every=20
save_every=10
evaluate_every=1

reg_smooth='none'
reg_lambda=0

seq_len=30
predict=1
pred_seq_len=30
predict_k=0

annot_loc_format='/sailhome/ajarno/STR-PIP/datasets/annot_{}_loc.pkl'

load_cache='feats'
# cache_format='/sailhome/bingbin/STR-PIP/datasets/cache/JAAD_conv_feats/concat_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt4_posNone_branchboth_collapse0_combinepair_cacheMasks_fixGRU_eval3_9acts_noAct_sanityWithPose_withReLU_pedGRU/{}/ped{}_fid{}.pkl'
cache_format='/sailhome/bingbin/STR-PIP/datasets/cache/jaad_loc/JAAD_conv_feats/loc_concat_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt1_posNone_branchboth_collapse0_combinepair_tmp/{}/vid{}_fid{}.pkl'

use_gru=1
use_trn=0
frame_gru=1
node_gru=0

# features
use_act=0
use_gt_act=0
use_driver=0
use_pose=0
pos_mode='none'
# branch='ped'
branch='both'
# adj_type='spatial'
# adj_type='random'
# adj_type='uniform'
# adj_type='spatialOnly'
# adj_type='all'
adj_type='inner'
use_obj_cls=0
n_layers=2
diff_layer_weight=0
collapse_cls=0
combine_method='pair'

# saving & loading
# suffix='_v4Feats_pedGRU_newCtxtGRU_3evalEpoch'
suffix='_v4Feats_pedGRU_3evalEpoch'

if [[ "$node_gru" == 1 ]]
then
  suffix=$suffix"_nodeGRU"
fi

if [[ "$use_obj_cls" == 1 ]]
then
  suffix=$suffix"_objCls"
fi

if [[ "$use_driver" == 1 ]]
then
  suffix=$suffix"_useDriver"
fi

ckpt_name='branch'$branch'_collapse'$collapse_cls'_combine'$combine_method'_adjType'$adj_type'_nLayers'$n_layers'_diffW'$diff_layer_weight$suffix
# ckpt_name='graph_seq30_layer2_embed'

CUDA_VISIBLE_DEVICES=$gpu_id python3 -m train.py \
  --model='loc_graph' \
  --reg-smooth=$reg_smooth \
  --reg-lambda=$reg_lambda \
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
  --log-every=$log_every \
  --n-workers=$n_workers \
  --annot-loc-format=$annot_loc_format \
  --load-cache=$load_cache \
  --cache-format=$cache_format \
  --branch=$branch \
  --adj-type=$adj_type \
  --use-obj-cls=$use_obj_cls \
  --n-layers=$n_layers \
  --diff-layer-weight=$diff_layer_weight \
  --collapse-cls=$collapse_cls \
  --combine-method=$combine_method \
  --use-gru=$use_gru \
  --use-trn=$use_trn \
  --frame-gru=$frame_gru \
  --node-gru=$node_gru \
  --use-act=$use_act \
  --use-gt-act=$use_gt_act \
  --use-driver=$use_driver \
  --use-pose=$use_pose \
  --pos-mode=$pos_mode
