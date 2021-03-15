import os
import pickle
import numpy as np

H, W = 1080, 1920
k = 0.25
c = -20
y_max = 200

def parse_split(fannot_in, fannot_out):
  with open(fannot_in, 'rb') as handle:
    peds = pickle.load(handle)

  vids = {}
  for ped in peds:
    vid = ped['vid']
    if vid not in vids:
      vids[vid] = {
        'act': np.zeros([len(ped['act']), 1]),
        'ped_pos': [[] for _ in range(len(ped['act']))]
      }
    for fid, pos in enumerate(ped['pos_GT']):
      if len(pos) == 0:
        continue
      x, y, w, h = pos
      cx, cy = x+0.5*w, y+h
      # check if a pedestrian is in the trapezoid area
      if H-cy <= k*cx+c and H-cy <= y_max and H-cy <= k*(W-cx)+c:
        vids[vid]['act'][fid] = 1
      vids[vid]['ped_pos'][fid] += pos,
  
  with open(fannot_out, 'wb') as handle:
    pickle.dump(vids, handle)


def parse_split_wrapper():
  annot_root = '/sailhome/agalczak/crossing/datasets'

  # ftrain = 'annot_train_ped_withTag_sanityWithPose.pkl'
  # annot_train = os.path.join(annot_root, ftrain)
  # annot_train_out = os.path.join(annot_root, 'annot_train_loc.pkl')
  # parse_split(annot_train, annot_train_out)

  # ftest = 'annot_test_ped_withTag_sanityWithPose.pkl'
  ftest = 'annot_test_ped.pkl'
  annot_test = os.path.join(annot_root, ftest)
  annot_test_out = os.path.join(annot_root, 'annot_test_loc_new.pkl')
  parse_split(annot_test, annot_test_out)


if __name__ == '__main__':
  parse_split_wrapper()
