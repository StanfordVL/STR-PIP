import os
import numpy as np
import pickle

import pdb

def smooth_vote(fpkl, win_size=5):
  with open(fpkl, 'rb') as handle:
    data = pickle.load(handle)
  out = data['out']
  gt = data['GT']
  acc = (out == gt).mean()

  smooth = np.zeros_like(out)
  win_half = win_size // 2
  smooth[:, :win_half] = out[:, :win_half]

  z2o, o2z = 0, 0
  for i,row in enumerate(out):
    for j in range(win_half, len(row)):
      # vote within the window
      smooth[i, j] = 1 if row[j-win_half:j+win_half].mean() >= 0.5 else 0
      if smooth[i,j] != out[i,j]:
        if out[i,j] == 0:
          z2o += 1
        else:
          o2z += 1
  acc_smooth = (smooth == gt).mean()

  print('Acc:', acc)
  print('Acc smooth:', acc_smooth)

  print('o2z:', o2z)
  print('z2o:', z2o)

def smooth_flip(fpkl):
  with open(fpkl, 'rb') as handle:
    data = pickle.load(handle)
  out = data['out']
  gt = data['GT']
  acc = (out == gt).mean()

  smooth = np.zeros_like(out)
  smooth[:, :1] = out[:, :1]

  z2o, o2z = 0, 0
  for i,row in enumerate(out):
    for j in range(1, len(row)-1):
      # check with the neighbors
      if row[j] != row[j-1] and row[j] != row[j+1]:
        smooth[i,j] = row[j-1]
        if row[j-1] == 0:
          o2z += 1
        else:
          z2o += 1
      else:
        smooth[i,j] = row[j]
  acc_smooth = (smooth == gt).mean()

  print('Acc:', acc)
  print('Acc smooth:', acc_smooth)

  print('o2z:', o2z)
  print('z2o:', z2o)

def smooth_sticky(fpkl, win_size=5):
  with open(fpkl, 'rb') as handle:
    data = pickle.load(handle)
  out = data['out']
  gt = data['GT']
  acc = (out == gt).mean()

  smooth = np.zeros_like(out)
  smooth[:, :win_size] = out[:, :win_size]

  for i,row in enumerate(out):
    for j in range(win_size, len(row)):
      if smooth[i, j-win_size:j].mean() > 0.5:
        smooth[i, j] = 1
      else:
        smooth[i,j] = row[j]
  acc_smooth = (smooth == gt).mean()

  print('Acc:', acc)
  print('Acc smooth:', acc_smooth)

  o2z = ((smooth!=out) * (out==1)).sum()
  z2o = ((smooth!=out) * (out==0)).sum()
  print('o2z:', o2z)
  print('z2o:', z2o)
  
  print('GT o:', gt.mean())
  print('GT nelem:', gt.size)

def smooth_hold(fpkl):
  # Stay 1 once gets to 1.
  with open(fpkl, 'rb') as handle:
    data = pickle.load(handle)
  out = data['out'][:, :-1]
  gt = data['GT'][:, :-1]
  print('Acc det:', (out[:, :30] == gt[:, :30]).mean())

  out = out[:, 30:]
  gt = gt[:, 30:]
  acc = (out == gt).mean()

  # pdb.set_trace()

  smooth = out.copy()
  for i,row in enumerate(smooth):
    # pdb.set_trace()
    pos = np.where(row == 1)[0]
    if len(pos):
      first = pos[0]
      row[first:] = 1
  acc_smooth = (smooth == gt).mean()

  print('Acc pred:', acc)
  print('Acc pred smooth:', acc_smooth)
  print('Acc last:', (smooth[:, -1] == gt[:, -1]).mean())

def smooth_hold_lastObserve(fpkl):
  # Stay 1 once gets to 1.
  with open(fpkl, 'rb') as handle:
    data = pickle.load(handle)
  out = data['out'][:, :-1]
  gt = data['GT'][:, :-1]
  print('Acc det:', (out[:, :30] == gt[:, :30]).mean())

  last_observe = out[:, 29]
  print('prob of last==1:', last_observe.mean())
  out = out[:, 30:]
  gt = gt[:, 30:]
  acc = (out == gt).mean()

  # pdb.set_trace()

  smooth = out.copy()
  for i,row in enumerate(smooth):
    if last_observe[i] == 1:
      row[:] = 1
  acc_smooth = (smooth == gt).mean()

  print('Acc pred:', acc)
  print('Acc pred smooth:', acc_smooth)
  print('Acc last:', (smooth[:, -1] == gt[:, -1]).mean())


def smooth_wrapper():
  # ped-centric model
  ckpt_dir = '/sailhome/bingbin/crossing/ckpts/JAAD/'
  # ckpt_name = 'graph_gru_seq30_pred30_lr1.0e-05_wd1.0e-05_bt16_posNone_branchped_collapse0_combinepair_adjTypeembed_nLayers2_v2Feats'
  # label_name = 'label_epochbest_det_stepall.pkl'
  ckpt_name = 'graph_gru_seq30_pred30_lr3.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_v4Feats_pedGRU_3evalEpoch'
  label_name = 'label_epochbest_pred_stepall.pkl'

  fpkl = os.path.join(ckpt_dir, ckpt_name, label_name)

  win_size = 3
  print('Vote (win_size={}):'.format(win_size))
  smooth_vote(fpkl, win_size=win_size)

  print('\nFlip:')
  smooth_flip(fpkl)

  print('\nSticky:')
  smooth_sticky(fpkl)

  print('\nHold:')
  smooth_hold(fpkl)

  print('\nHold last observed:')
  smooth_hold_lastObserve(fpkl)

if __name__ == '__main__':
  smooth_wrapper()
