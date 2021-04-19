import os
import pdb
import pickle
from glob import glob

data_root = '/sailhome/ajarno/STR-PIP/datasets/'
img_root = os.path.join(data_root, 'JAAD_dataset/JAAD_clip_images')

def count_frames():
  vids = glob(os.path.join(img_root, 'video_*.mp4'))
  cnts = []
  for vid in vids:
    cnt = len(glob(os.path.join(vid, '*.jpg')))
    cnts += cnt,

  print('total # frames:', sum(cnts))
  print('cnts: max={} / min={} / mean={}'.format(max(cnts), min(cnts), sum(cnts)/len(cnts)))

def ped_count_crossing():
  def helper(ped):
    act = ped['act']
    crossing = [each[1] for each in act]
    n_cross = sum(crossing)
    n_noncross = len(crossing) - n_cross
    return n_cross, n_noncross

  with open(os.path.join(data_root, 'annot_train_ped_withTag_sanityWithPose.pkl'), 'rb') as handle:
    train = pickle.load(handle)
  with open(os.path.join(data_root, 'annot_test_ped_withTag_sanityWithPose.pkl'), 'rb') as handle:
    test = pickle.load(handle)

  n_crosses = 0
  n_noncrosses = 0
  for ped in train+test:
    n_cross, n_noncross = helper(ped)
    n_crosses += n_cross
    n_noncrosses += n_noncross
  
  n_total = n_crosses + n_noncrosses
  print('n_cross: {} ({:.4f})'.format(n_crosses, n_crosses/n_total))
  print('n_noncross: {} ({:.4f})'.format(n_noncrosses, n_noncrosses/n_total))

def loc_count_crossing():
  def helper(split):
    split_cross, split_noncross = 0, 0
    for vid in split:
      n_cross = sum(split[vid]['act'] == 1)[0]
      n_noncross = sum(split[vid]['act'] == 0)[0]
      split_cross += n_cross
      split_noncross += n_noncross
    split_total = split_cross + split_noncross
    # pdb.set_trace()
    print('  cross:{} ({:.4f}) / non-cross:{} ({:.4f}) / total:{}'.format(
      split_cross, split_cross / split_total, split_noncross, split_noncross / split_total, split_total))
    return split_cross, split_noncross, split_total

  with open(os.path.join(data_root, 'annot_train_loc.pkl'), 'rb') as handle:
    train = pickle.load(handle)
  print('train:')
  train_cross, train_noncross, train_total = helper(train)

  with open(os.path.join(data_root, 'annot_test_loc.pkl'), 'rb') as handle:
    test = pickle.load(handle)
  print('\ntest:')
  test_cross, test_noncross, test_total = helper(test)

  n_cross, n_noncross, n_total = train_cross+test_cross, train_noncross+test_noncross, train_total+test_total
  print('\ntotal:')
  print('  cross:{} ({:.4f}) / non-cross:{} ({:.4f}) / total:{}'.format(
    n_cross, n_cross / n_total, n_noncross, n_noncross / n_total, n_total))

def loc_count_ped():
  def helper(split):
    n_peds = []
    for vid in split:
      for frame in split[vid]['ped_pos']:
        n_peds += len(frame),
    print('  max:{} / min:{} / mean:{}'.format(max(n_peds), min(n_peds), sum(n_peds)/len(n_peds)))
    return n_peds

  with open(os.path.join(data_root, 'annot_train_loc.pkl'), 'rb') as handle:
    train = pickle.load(handle)
  print('train:')
  train_n_peds = helper(train)

  with open(os.path.join(data_root, 'annot_test_loc_new.pkl'), 'rb') as handle:
    test = pickle.load(handle)
  print('\ntest:')
  test_n_peds = helper(test)

  n_peds = train_n_peds + test_n_peds
  print('\ntotal:')
  print('  max:{} / min:{} / mean:{}'.format(max(n_peds), min(n_peds), sum(n_peds)/len(n_peds)))


if __name__ == '__main__':
  print('count_frames:')
  count_frames()

  print('\nped_count_crossing:')
  ped_count_crossing()

  print('\nloc_count_crossing:')
  loc_count_crossing()

  print('\nloc_count_ped:')
  loc_count_ped()
