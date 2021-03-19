import os
import numpy as np
import pickle

def find_segments(labels):
  durs = []
  for row in labels:
    s_start = 0
    for j,val in enumerate(row):
      if j and row[j-1] != val:
        durs += j-s_start,
        s_start = j
    durs += len(row) - s_start,
  print('# segments:', len(durs))
  print('# avg frames:', np.mean(durs))

def find_segments_wrapper():
  ckpt_dir = '/sailhome/bingbin/STR-PIP/ckpts/JAAD/'
  ckpt_name = 'graph_gru_seq30_pred30_lr1.0e-05_wd1.0e-05_bt16_posNone_branchped_collapse0_combinepair_adjTypeembed_nLayers2_v2Feats'
  label_name = 'label_epochbest_det_stepall.pkl'
  fpkl = os.path.join(ckpt_dir, ckpt_name, label_name)

  with open(fpkl, 'rb') as handle:
    data = pickle.load(handle)
  out = data['out']
  gt = data['GT']

  print('Out:')
  find_segments(out)
  print('GT:')
  find_segments(gt)

  print('\nOut det:')
  find_segments(out[:, :30])
  print('\nOut pred:')
  find_segments(out[:, 30:])
  print('\nGT det:')
  find_segments(gt[:, :30])
  print('\nGT pred:')
  find_segments(gt[:, 30:])
  
if __name__ == '__main__':
  find_segments_wrapper()
