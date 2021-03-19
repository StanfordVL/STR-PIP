#!/bin/bash

mode='evaluate'
split='test'

gpu_id=1
n_workers=0
n_acts=9
bt=1

seq_len=30
predict=1
pred_seq_len=30
predict_k=0

# test only
slide=0
rand_test=1
log_every=10
ckpt_dir='/sailhome/bingbin/STR-PIP/ckpts'
dataset='JAAD'

# pred 30
# ckpt_name='graph_gru_seq30_pred30_lr1.0e-05_wd1.0e-05_bt16_posNone_branchped_collapse0_combinepair_adjTypeembed_nLayers2_v2Feats'
# ckpt_name='graph_gru_seq30_pred30_lr3.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_v4Feats_pedGRU_3evalEpoch'
# ckpt_name='graph_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_diffW1_v4Feats_pedGRU_3evalEpoch'
# ckpt_name='graph_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_v4Feats_pedGRU_3evalEpoch'
ckpt_name='graph_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_v4Feats_pedGRU_newCtxtGRU_3evalEpoch' # best

# pred 60
# ckpt_name='graph_gru_seq30_pred60_lr1.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_diffW0_v4Feats_pedGRU_3evalEpoch'

# pred 90
# ckpt_name='graph_gru_seq30_pred90_lr1.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_diffW0_v4Feats_pedGRU_3evalEpoch'

which_epoch=-1
if [ $which_epoch -eq -1 ]
then
  epoch_name='best_pred'
else
  epoch_name=$which_epoch
fi
save_output=1
save_output_format=$ckpt_dir'/'$dataset'/'$ckpt_name'/output_epoch'$epoch_name'_step{}.pkl'
collect_A=1
save_As_format=$ckpt_dir'/'$dataset'/'$ckpt_name'/test_graph_weights_epoch'$epoch_name'/vid{}_eval{}.pkl'



load_cache='feats'
# cache_format='/sailhome/bingbin/STR-PIP/datasets/cache/JAAD_conv_feats/concat_gru_lr1.0e-04_wd1.0e-05_bt4_ped_collapse0_combinepair_useBBox0_cacheMasks_fixGRU_singleTime/{}/ped{}_fid{}.pkl'
cache_format='/sailhome/bingbin/STR-PIP/datasets/cache/JAAD_conv_feats/concat_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt4_posNone_branchboth_collapse0_combinepair_cacheMasks_fixGRU_eval3_9acts_noAct_sanityWithPose_withReLU_pedGRU/{}/ped{}_fid{}.pkl'


# cache_format='/sailhome/bingbin/STR-PIP/datasets/cache/JAAD_conv_feats/concat_gru_lr1.0e-05_bt4_test_epoch5/{}/ped{}_fid{}.pkl'

# ckpt_name='graph_gru_lr1.0e-05_wd1.0e-05_bt16_ped_collapse0_combinepair_adjTypeembed_nLayers0_useBBox0_fixGRU'
# which_epoch=50
# 
# ckpt_name='graph_gru_lr1.0e-05_wd1.0e-05_bt16_ped_collapse0_combinepair_adjTypeembed_nLayers2_useBBox0_fixGRU'
# which_epoch=22
# 
# ckpt_name='graph_gru_lr1.0e-05_wd1.0e-05_bt16_both_collapse0_combinepair_adjTypeembed_nLayers2_useBBox0_fixGRU'
# which_epoch=38
# 
# ckpt_name='graph_gru_lr1.0e-05_wd1.0e-05_bt16_ped_collapse0_combinepair_adjTypeuniform_nLayers0_useBBox0_fixGRU'
# which_epoch=42

if [ "$mode" = "extract" ]
then
  extract_feats_dir='/sailhome/bingbin/STR-PIP/datasets/cache/JAAD_conv_feats/concat_gru_lr1.0e-05_bt4_test_epoch5/test/'
else
  extract_feats_dir='none_existent'
fi

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
use_obj_cls=0
n_layers=2
diff_layer_weight=0
collapse_cls=0
combine_method='pair'

CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py \
  --model='graph' \
  --split=$split \
  --mode=$mode \
  --slide=$slide \
  --rand-test=$rand_test \
  --ckpt-dir=$ckpt_dir \
  --ckpt-name=$ckpt_name \
  --n-acts=$n_acts \
  --device=$gpu_id \
  --dset-name='JAAD' \
  --ckpt-name=$ckpt_name \
  --which-epoch=$which_epoch \
  --save-output=$save_output \
  --save-output-format=$save_output_format \
  --collect-A=$collect_A \
  --save-As-format=$save_As_format \
  --n-epochs=$n_epochs \
  --start-epoch=$start_epoch \
  --seq-len=$seq_len \
  --predict=$predict \
  --pred-seq-len=$pred_seq_len \
  --predict-k=$predict_k \
  --batch-size=$bt \
  --log-every=$log_every \
  --n-workers=$n_workers \
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

exit

# bak of prev command

CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py \
  --model='graph' \
  --split=$split \
  --mode=$mode \
  --device=$gpu_id \
  --log-every=$log_every \
  --dset-name=$dataset \
  --ckpt-dir=$ckpt_dir \
  --ckpt-name=$ckpt_name \
  --batch-size=1 \
  --n-workers=$n_workers \
  --load-cache=$load_cache \
  --cache-format=$cache_format \
  --n-layers=$n_layers \
  --seq-len=$seq_len \
  --predict=$predict \
  --pred-seq-len=$pred_seq_len \
  --use-gru=$use_gru \
  --pos-mode=$pos_mode \
  --adj-type=$adj_type \
  --collapse-cls=$collapse_cls \
  --slide=$slide \
  --rand-test=$rand_test \
  --branch=$branch \
  --which-epoch=$which_epoch \
  --extract-feats-dir=$extract_feats_dir \
  --save-output=$save_output \
  --save-output-format=$save_output_format

