import os
import torch

ckpt_root = '/sailhome/bingbin/STR-PIP/ckpts/'
dataset = 'JAAD'

def check(ckpt, ptype='pred'):
  f_best_pred = os.path.join(ckpt_root, dataset, ckpt, 'best_{}.pth'.format(ptype))
  pred = torch.load(f_best_pred)
  print('best {} epoch:'.format(ptype), pred['epoch'])

if __name__ == '__main__':
  ckpt = 'graph_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_v4Feats_pedGRU_3evalEpoch'
  check(ckpt, 'pred')
  check(ckpt, 'last')
  check(ckpt, 'det')
