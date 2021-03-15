#!/bin/bash

gpu_id=0
split='train'
n_workers=0
n_acts=1
bt=32
log_every=30

# Training only
n_epochs=200
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
pred_seq_len=60
predict_k=0

annot_ped_format='/vision/group/prolix/processed/pedestrians/{}.pkl'

cache_obj_bbox_format='/vision/group/prolix/processed/obj_bbox_20fps_merged/{}_seg{}.pkl'

load_cache='feats'
cache_format_base='/vision/group/prolix/processed/cache/STIP_conv_feats/'
cache_format_pkl='{}/ped{}_fid{}.pkl'
# cache_format=$cache_format_base'concat_gru_seq8_pred2_lr1.0e-04_wd1.0e-05_bt2_posNone_branchboth_collapse0_combinepair_testANN_hanh1/'$cache_format_pkl
# concat_ckpt='concat_gru_seq8_pred2_lr1.0e-04_wd1.0e-05_bt2_posNone_branchboth_collapse0_combinepair_run1/'
concat_ckpt='concat_gru_seq8_pred2_lr1.0e-05_wd1.0e-05_bt2_posNone_branchboth_collapse0_combinepair_decay10/'
cache_format=$cache_format_base$concat_ckpt$cache_format_pkl

use_gru=1
use_trn=0
ped_gru=1
ctxt_gru=0
ctxt_node=0

# features
use_act=0
use_gt_act=0
use_driver=0
use_pose=0
pos_mode='none'
# branch='ped'
branch='both'
adj_type='spatial'
# adj_type='random'
# adj_type='uniform'
# adj_type='spatialOnly'
# adj_type='all'
# adj_type='inner'
use_obj_cls=0
n_layers=2
diff_layer_weight=0
collapse_cls=0
combine_method='pair'

# saving & loading
suffix='_decay10Feats'

if [[ "$ctxt_gru" == 1 ]]
then
  suffix=$suffix"_newCtxtGRU"
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

# WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$gpu_id python3 -m train.py \
CUDA_VISIBLE_DEVICES=$gpu_id python3 -m train.py \
  --model='graph' \
  --reg-smooth=$reg_smooth \
  --reg-lambda=$reg_lambda \
  --split=$split \
  --n-acts=$n_acts \
  --device=$gpu_id \
  --dset-name='STIP' \
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
  --annot-ped-format=$annot_ped_format \
  --cache-obj-bbox-format=$cache_obj_bbox_format \
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
  --ped-gru=$ped_gru \
  --ctxt-gru=$ctxt_gru \
  --ctxt-node=$ctxt_node \
  --use-act=$use_act \
  --use-gt-act=$use_gt_act \
  --use-driver=$use_driver \
  --use-pose=$use_pose \
  --pos-mode=$pos_mode
