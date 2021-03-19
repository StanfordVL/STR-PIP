import os
from glob import glob
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

SMOOTH = True
MIX = True

fpkls = []

ckpt_root = '/sailhome/bingbin/STR-PIP/ckpts/JAAD/'

# best proposed

# pred 30

# ckpt_name = 'graph_gru_seq30_pred30_lr3.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_v4Feats_pedGRU_3evalEpoch'
# fout = 'label_epochbest_pred_stepall_run2.pkl'
# fpkl = os.path.join(ckpt_root, ckpt_name, fout)
# fpkls += fpkl,

fout = 'output_epochbest_pred_stepall.pkl'
# fpkl = os.path.join(ckpt_root, ckpt_name, fout)
# fpkls += fpkl,

ckpt_name = 'graph_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_v4Feats_pedGRU_3evalEpoch'
fpkl = os.path.join(ckpt_root, ckpt_name, fout)
# fpkls += fpkl,

# fout = 'output_epochbest_pred_stepall_run3Epochs.pkl'
# fout = 'output_epochbest_pred_stepall_run1Epochs.pkl'
# fpkl = os.path.join(ckpt_root, ckpt_name, fout)
# fpkls += fpkl,

ckpt_name = 'graph_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_v4Feats_pedGRU_newCtxtGRU_3evalEpoch'
fpkl = os.path.join(ckpt_root, ckpt_name, fout)
fpkls += fpkl,

# graph - pred 60
ckpt_name = 'graph_gru_seq30_pred60_lr1.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_diffW0_v4Feats_pedGRU_3evalEpoch'
fpkl = os.path.join(ckpt_root, ckpt_name, fout)
fpkls += fpkl,


# graph - pred 90
ckpt_name = 'graph_gru_seq30_pred90_lr1.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_diffW0_v4Feats_pedGRU_3evalEpoch'
fpkl = os.path.join(ckpt_root, ckpt_name, fout)
fpkls += fpkl,


# best concat (pred 30)
ckpt_name = 'concat_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt4_posNone_branchboth_collapse0_combinepair_cacheMasks_fixGRU_eval3_9acts_noAct_sanityWithPose_withReLU_pedGRU'
fout = 'output_all.pkl'
fpkl = os.path.join(ckpt_root, ckpt_name, fout)

# fpkls += fpkl,


for i, fpkl in enumerate(fpkls):
  with open(fpkl, 'rb') as handle:
    data = pickle.load(handle)
  out, gt = data['out'], data['GT']
  if out.shape[-1] % 10 == 1:
    out = out[:, :-1]
    gt = gt[:, :-1]
  try:
    acc = (out == gt).mean(0)
    pred_acc = (out[:, 30:] == gt[:, 30:]).mean()
    print('pred_acc:', pred_acc)
  except Exception as e:
    print(e)
    pdb.set_trace()
  acc = list(acc)
  if SMOOTH:
    t1, t3 = acc[1:]+[acc[-1]], [acc[0]]+acc[:-1]
    acc = (np.array(t1) + np.array(t3) + np.array(acc)) / 3
    # pdb.set_trace()
  plt.plot(acc, label='{} frames{}'.format(30*(i+1), ' (acc)' if MIX else ''))
plt.legend()
plt.xlabel('Frames', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.savefig('acc_temp_graph{}.png'.format('_smooth' if SMOOTH else ''), dpi=1000)
if not MIX:
  plt.clf()


for i, fpkl in enumerate(fpkls):
  with open(fpkl, 'rb') as handle:
    data = pickle.load(handle)
  prob = data['prob']
  B = data['out'].shape[0]
  prob = prob.reshape(B, -1)
  if prob.shape[-1] % 10 == 1:
    prob = prob[:, :-1]
  prob = list(prob.mean(0))
  if SMOOTH:
    t1, t3 = prob[1:]+[prob[-1]], [prob[0]]+prob[:-1]
    prob = (np.array(t1) + np.array(t3) + np.array(prob)) / 3
    # pdb.set_trace()
  plt.plot(prob, label='{} frames{}'.format(30*(i+1), ' (prob)' if MIX else ''))
plt.legend()
plt.xlabel('Frames', fontsize=12)
plt.ylabel('Accuracy & Probability' if MIX else 'Probability', fontsize=12)
plt.savefig('{}_temp_graph{}.png'.format('mix' if MIX else 'prob', '_smooth' if SMOOTH else ''), dpi=1000)
plt.clf()

