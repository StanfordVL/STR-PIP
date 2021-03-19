#!/bin/bash

mode='evaluate'
split='test'

gpu_id=1
n_workers=0
n_acts=1
bt=1

seq_len=4
predict=1
pred_seq_len=4
predict_k=0

view='all'

# TODO: get new ped formats
annot_ped_format='/vision/group/prolix/processed/pedestrians_all_views/{}.pkl'
cache_obj_bbox_format='/vision/group/prolix/processed/'$view'/obj_bbox_20fps_merged/{}_seg{}.pkl'

# test only
slide=0
rand_test=1
log_every=10
ckpt_dir='/sailhome/bingbin/STR-PIP/ckpts'
dataset='STIP'

if [ $seq_len -eq 8 ]
then
  ckpt_name='graph_gru_seq'$seq_len'_pred'$pred_seq_len'_lr1.0e-04_wd1.0e-05_bt32_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_diffW0_decay10Feats_all'
else
  # seq_len = 4
  ckpt_name='graph_gru_seq'$seq_len'_pred'$pred_seq_len'_lr1.0e-04_wd1.0e-05_bt32_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_diffW0_decay10Feats'
fi
which_epoch=-1
if [ $which_epoch -eq -1 ]
then
  epoch_name='best_pred'
else
  epoch_name=$which_epoch
fi
save_output=1
save_output_format=$ckpt_dir'/'$dataset'/'$ckpt_name'/output_epoch'$epoch_name'_step{}_'$seq_len'+'$pred_seq_len'.pkl'
collect_A=1
save_As_format=$ckpt_dir'/'$dataset'/'$ckpt_name'/test_graph_weights_epoch'$epoch_name'/vid{}_eval{}.pkl'


load_cache='feats'
feat_ckpt_name='concat_gru_seq8_pred2_lr1.0e-05_wd1.0e-05_bt2_posNone_branchboth_collapse0_combinepair_decay10'
cache_format='/vision/group/prolix/processed/cache/'$view'/STIP_conv_feats/'$feat_ckpt_name'/{}/ped{}_fid{}.pkl'


# if [ "$mode" = "extract" ]
# then
#   extract_feats_dir='/sailhome/bingbin/STR-PIP/datasets/cache/JAAD_conv_feats/concat_gru_lr1.0e-05_bt4_test_epoch5/test/'
# else
#   extract_feats_dir='none_existent'
# fi

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

CUDA_VISIBLE_DEVICES=$gpu_id python3 -W ignore test.py \
  --model='graph' \
  --split=$split \
  --view=$view \
  --mode=$mode \
  --slide=$slide \
  --rand-test=$rand_test \
  --ckpt-dir=$ckpt_dir \
  --ckpt-name=$ckpt_name \
  --n-acts=$n_acts \
  --device=$gpu_id \
  --dset-name='STIP' \
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


