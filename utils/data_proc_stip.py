import os
from glob import glob
import numpy as np
import pickle
import cv2
import json
import xml.etree.ElementTree as ET
from pycocotools import mask as maskUtils

import pdb

ALL_VIEW = True
all_view_skip_list = ['downtown_ann_1-09-26-2017', 'downtown_ann_2-09-28-2017', 'downtown_ann_3-09-28-2017']

data_root = '/vision/group/prolix'
# input
# annot_root = os.path.join(data_root, 'annotations')
annot_root = '/sailhome/agalczak/crossing/datasets/STIP/annotations'
fps20_frames_root = os.path.join(data_root, 'images_20fps')
fps20_instances_root = os.path.join(data_root, 'instances_20fps/stip_instances/')
# map12to20_root = os.path.join(data_root, 'mapping_12_to_20')
map12to20_root = os.path.join('/cvgl2/u/ashenoi/STIP/', 'mapping_12_to_20fps')
# output
out_root = os.path.join(data_root, 'processed')
os.makedirs(out_root, exist_ok=True)

if ALL_VIEW:
  # NOTE: changed for "all_views"
  out_ped_root = os.path.join(out_root, 'pedestrians_all_views')
  fps20_instances_root_left = os.path.join(data_root, 'stip_side/left/instances_20fps/')
else:
  out_ped_root = os.path.join(out_root, 'pedestrians')
os.makedirs(out_ped_root, exist_ok=True)

# Objects of interest
cls_map = {
  # other road users (treated as part of vehicles)
  5: 1, # bicyclist
  6: 1, # motorcyclist
  7: 1, # other riders
  # vehicles
  28: 1, # bicycle
  29: 1, # boat (??)
  30: 1, # bus
  31: 1, # car
  32: 1, # caravan
  33: 1, # motorcycle
  34: 1, # other vehicle
  35: 1, # trailer
  36: 1, # truck
  37: 1, # wheeled slow
  # environments
  3: 2, # crosswalk - plain
  8: 3, # crosswalk - zebra
  24: 4, # traffic lights
}

# smaller set of objs becaues of memory
cls_map_small = {
  # vehicles
  28: 1, # bicycle
  30: 1, # bus
  31: 1, # car
  33: 1, # motorcycle
  35: 1, # trailer
  36: 1, # truck
  # environments
  3: 2, # crosswalk - plain
  8: 3, # crosswalk - zebra
  24: 4, # traffic lights
}


def parse_objs(fnpy):
  # Parse instance segmentations on one frame to masks.
  # In: filename of instance segmentation (i.e. `dataset/JAAD_instance_segm/video_{:04d}/{:08d}_segm.npy`)
  # Out: dict with key 1-4 (see `cls_map`), each for a type of graph nodes.
  segms = np.load(fnpy)

  selected_segms = {}
  for key,val in cls_map.items():
    if segms[key]:
      if val not in selected_segms:
        selected_segms[val] = []
      for each in segms[key]:
        mask = maskUtils.decode([each])
        selected_segms[val] += mask,
  del segms
  return selected_segms

def stip_segm2box(fsegm):
  segms = np.load(fsegm)
  boxes = []
  for cls_segms in segms:
    # decode each 'count' to a ndarray of the size of the image
    cls_segms = [maskUtils.decode([each]) for each in cls_segms]

    cls_boxes = []
    for mask in cls_segms:
      y_pos = mask.sum(1).nonzero()[0]
      x_pos = mask.sum(0).nonzero()[0]
      if len(y_pos) == 0:
        print('empty y_pos:', fnpy)
        continue
      if len(x_pos) == 0:
        print('empty x_pos:', fnpy)
        continue
      y_min, y_max = y_pos[0], y_pos[-1]
      x_min, x_max = x_pos[0], x_pos[-1]
      box = [x_min, y_min, x_max, y_max]
      cls_boxes += np.array(box),
    boxes += np.array(cls_boxes),
  fbox = fsegm.replace('segm', 'box')
  assert(fbox != fsegm)
  np.save(fbox, np.array(boxes))

def stip_segm2box_wrapper():
  # Parse encoded masks to box coordinates
  inst_root = fps20_instances_root
  for vi, vid in enumerate(sorted(glob(os.path.join(inst_root, '*_*')))):
    inst_dir = os.path.join(vid, 'inference')
    if not os.path.exists(inst_dir):
      continue
    print(os.path.basename(os.path.dirname(inst_dir)))
    vid_segs = sorted(glob(os.path.join(inst_dir, '*--*')))
    for vid_seg in vid_segs:
      print('\t' + os.path.basename(vid_seg))
      fsegms = sorted(glob(os.path.join(vid_seg, '*segm.npy')))
      for fsegm in fsegms:
        stip_segm2box(fsegm)


def get_obj_crops(fnpy_root, fpkl_root):
  os.makedirs(fpkl_root, exist_ok=True)

  vids = sorted(glob(os.path.join(fnpy_root, 'video_*')))
  for i,vid in enumerate(vids):
    if i < 2:
      continue
    print('vid:', os.path.basename(vid))
    fnpys = sorted(glob(os.path.join(vid, '*_segm.npy')))
    for fnpy in fnpys:
      selected_segms = parse_objs(fnpy)
      crops = {}
      for cls,segms in selected_segms.items():
        if len(segms):
          crops[cls] = []
          for mask in segms:
            y_pos = mask.sum(1).nonzero()[0]
            x_pos = mask.sum(0).nonzero()[0]
            if len(y_pos) == 0:
              print('empty y_pos:', fnpy)
              continue
            if len(x_pos) == 0:
              print('empty x_pos:', fnpy)
              continue
            y_min, y_max = y_pos[0], y_pos[-1]
            x_min, x_max = x_pos[0], x_pos[-1]
            if x_min >= x_max or y_min >= y_max:
              print('empty_crop:', fnpy)
              print('x_min={} / x_max={} / y_min={} / y_max={}\n'.format(x_min, x_max, y_min, y_max))
              continue
            crop = mask[y_min:(y_max+1), x_min:(x_max+1)]
            crop = cv2.resize(crop, (224,224))
            crops[cls] += crop,
          if len(crops[cls]):
            crops[cls] = np.stack(crops[cls])
          else:
            crops[cls] = np.zeros([0, 224, 224])
        else:
          crops[cls] = np.zeros([0, 224, 224])

      fpkl = fnpy.replace(fnpy_root, fpkl_root)
      fpkl_dir = os.path.dirname(fpkl)
      os.makedirs(fpkl_dir, exist_ok=True)
      with open(fpkl, 'wb') as handle:
        pickle.dump(crops, handle)


joint_ids = [
  1,  # 0 Neck
  2,  # 1 RShoulder
  5,  # 2 LShoulder
  9,  # 3 RHip
  10, # 4 RKnee
  11, # 5 RAnkle
  12, # 6 LHip
  13, # 7 LKnee
  14, # 8 LAnkle
]


def parse_pedestrian(fjson, vid_root, fmap12to20=None, fpos_GT='', fpos_pred='', fpose=''):
  """
  parse json files: /vision/group/prolix/annotations
  ret: dict w/ keys: 'act', 'pid', 'frame_start', 'frame_end', 'pos_GT', 'vid'
  (not included: 'pose', 'pos_Pose', 'occl', 'tag')
  """
  # NOTE: this is 2fps w/ GT annot ped bbox & no interpolation

  fout = open(os.path.join(out_ped_root, 'log_todo.txt'), 'a')

  vid = os.path.basename(vid_root)

  if fmap12to20:
    with open(fmap12to20, 'rb') as handle:
      map12to20 = pickle.load(handle)

  ffid2seg = os.path.join(vid_root, 'fid2seg_allView.pkl' if ALL_VIEW else 'fid2seg.pkl')
  if not os.path.exists(ffid2seg) or True:
    # calculate the mapping from frame ids (fid) to segment names (seg)
    fid2seg = {}
    segs = sorted(glob(os.path.join(vid_root, 'inference', '*--*')))
    seg_start_cnt = 0
    for seg in segs:
      fids = [os.path.basename(each).split('_')[0] for each in sorted(glob(os.path.join(seg, '*_segm.npy')))]

      if ALL_VIEW:
        filtered_fids = []
        for fid in fids:
          left_fid = seg.replace(fps20_instances_root, fps20_instances_root_left).replace('inference/', '')
          left_fid = os.path.join(left_fid, '{}.pkl'.format(fid))
          if os.path.exists(left_fid):
            filtered_fids += fid,
        fids = filtered_fids
      seg = os.path.basename(seg)
      for fid in fids:
        fid2seg[fid] = {'seg': seg, 'seg_start':seg_start_cnt}
      seg_start_cnt += len(fids)
    with open(ffid2seg, 'wb') as handle:
      pickle.dump(fid2seg, handle)
  else:
    # load from pickle
    with open(ffid2seg, 'rb') as handle:
      fid2seg = pickle.load(handle)


  data = json.load(open(fjson, 'r'))
  frames = data['frames']
  frame_keys = sorted(frames.keys(), key=lambda x:int(x))
  print("# frames:", len(frames))


  seg2fids = {}
  for key in sorted(fid2seg):
    seg = fid2seg[key]['seg']
    if seg not in seg2fids:
      seg2fids[seg] = []
    seg2fids[seg] += key,

  peds = {}
  matchIds = [] # use list instead of set to tain the order in which pedestrians are added.
  for fid in frame_keys:
    if not frames[fid]:
      # skip frames w/o pedestrians
      continue

    if fmap12to20:
      # input is 2fps: need to get 2fps to 12fps then to 20fps
      # map to 12 fps
      fid_12fps = (int(fid)-1) * 6
      # fid_12fps_key = '{:010d}'.format(fid_12fps)
      fid_12fps_key = str(fid_12fps)

      if fid_12fps_key not in map12to20['mapping']:
        # only use frames marked in the segments
        continue
        # raise KeyError("No {} in {}".format(fid_12fps_key, os.path.basename(fmap12to20)))
        fid_20fps = -1
      else:
        # map to 20 fps
        # type(fid_20fps): int
        fid_20fps = map12to20['mapping'][fid_12fps_key]
    else:
      # input is 20fps
      fid_2fps = None
      fid_20fps = int(fid)


    fid_20fps_key = '{:010d}'.format(fid_20fps)
    if fid_20fps_key in fid2seg:
      seg = fid2seg[fid_20fps_key]['seg']
    else:
      continue # TODO: ignoring frames not in a segment for now.
      seg = "TODO"
      fout.write("{}\t{}\n".format(vid, fid))


    for ped in frames[fid]:
      # append seg to mid: handle each segment separately
      mid = str(ped['boxId']) + '_' + seg

      if mid not in matchIds:
        peds[mid] = {'act':[], 'pid':ped['boxId'], 'frame_start_2fps':int(fid), 'frame_start_20fps':fid_20fps, 'fids20':[],
                     'pos_GT':[], 'vid':os.path.basename(vid), 'seg':seg, 'mid':mid, 'fids12': [], 'fids2':[]}
        # mapping from 20fps fid to relative indices in a video segment
        peds[mid]['map20fpsToRel'] = {}
        matchIds += mid,

      peds[mid]['fids20'] += fid_20fps,     # int
      if fmap12to20:
        peds[mid]['fids12'] += fid_12fps_key, # str
        peds[mid]['fids2'] += fid,            # str
      peds[mid]['act'] += int(ped['crossed']),
      x1, x2, y1, y2 = ped['box']['x1'], ped['box']['x2'], ped['box']['y1'], ped['box']['y2']
      box = np.array([x1, y1, x2-x1, y2-y1]).astype(np.int64) # (x, y, w, h)
      peds[mid]['pos_GT'] += box,
      peds[mid]['frame_end_2fps'] = int(fid) # inclusive, i.e. pedestrian appears in 'frame_end'.
      peds[mid]['frame_end_20fps'] = fid_20fps
      # relative frame index in the segment
      # fid_rel_seg_idx = fid_sorted.index(fid) - fid2seg[fid_20fps_key]['seg_start']
      fid_rel_seg_idx = seg2fids[seg].index(fid_20fps_key)
      peds[mid]['map20fpsToRel'][fid_20fps] = fid_rel_seg_idx
  fout.close()
  print('#peds:', len(peds))
  return peds


def parse_pedestrian_wrapper():
  all_jsons = glob(os.path.join(annot_root, '*annotation.json'))

  vid_roots = sorted(glob(os.path.join(fps20_instances_root, '*_*')))
  print("# vids:", len(vid_roots))
  all_peds = []
  for vid_root in vid_roots:
    vid = os.path.basename(vid_root)
    if ALL_VIEW and vid in all_view_skip_list:
      continue

    # fjsons = [each for each in fjsons if vid.replace('-', '') in each.replace('-', '')]
    fjsons = [each for each in all_jsons if os.path.basename(vid.replace('-', '')) in each.replace('-', '')]
    print('{} ({} annots)'.format(vid, len(fjsons)))

    # TODO: I forgot why checking the len...
    # if len(fjsons) > 1:
    #   raise ValueError('len(fjson): {}'.format(len(fjson)))
    if len(fjsons) == 0:
      # No annotation json file
      print('Skipped:', vid)
      continue
    fjson = fjsons[0] # diff seges should have the same json file.

    # fmap12to20 = os.path.join(map12to20_root, "{}.p".format(vid))
    fmap12to20 = None
    fped_out = os.path.join(out_ped_root, '{}.pkl'.format(vid))
    if os.path.exists(fped_out):
      peds = pickle.load(open(fped_out, 'rb'))
    else:
      peds = parse_pedestrian(fjson, vid_root, fmap12to20, fpos_GT='', fpos_pred='', fpose='')

    for key in sorted(peds.keys()):
      all_peds += peds[key],
    with open(fped_out, 'wb') as handle:
      pickle.dump(peds, handle)

  with open(os.path.join(out_ped_root, 'all_peds.pkl'), 'wb') as handle:
    pickle.dump(all_peds, handle)

  return

  # TODO: prepare splits w/ all data


def prepare_split_wrapper():
  def prepare_split(fpeds):
    all_peds = []
    for fped in fpeds:
      with open(fped, 'rb') as handle:
        peds = pickle.load(handle)
      try:
        for key in sorted(peds):
          all_peds += peds[key],
      except Exception as e:
        pdb.set_trace()
    return all_peds

  all_files = sorted(glob(os.path.join(out_ped_root, '*pkl')))
  # skip pkl files that are themselves splits.
  all_files = [os.path.basename(each) for each in all_files]
  all_files = [each for each in all_files if 'train' not in each and 'test' not in each and 'all' not in each]

  # file names
  test_files = ['downtown_ann_3-09-28-2017.pkl', 'downtown_palo_alto_6.pkl', 'dt_san_jose_4.pkl', 'mountain_view_4.pkl', 'sf_soma_2.pkl']
  train_files = [each for each in all_files if each not in test_files]
  if ALL_VIEW:
    test_files = [each for each in test_files if each.split('.')[0] not in all_view_skip_list]
    train_files = [each for each in train_files if each.split('.')[0] not in all_view_skip_list]

  # append dir path
  train_files = [os.path.join(out_ped_root, train_file) for train_file in train_files]
  test_files = [os.path.join(out_ped_root, test_file) for test_file in test_files]

  # Train
  print('Processing training data...')
  annot_train = prepare_split(train_files)
  print('# Train:', len(annot_train))
  with open(os.path.join(out_ped_root, 'train.pkl'), 'wb') as handle:
    pickle.dump(annot_train, handle)
  print('Training data ready.\n')

  # Test
  print('Processing testing data...')
  annot_test = prepare_split(test_files)
  print('# Test:', len(annot_test))
  with open(os.path.join(out_ped_root, 'test.pkl'), 'wb') as handle:
    pickle.dump(annot_test, handle)
  print('Testing data ready.\n')

def fill_20fps():
  def helper(fpkl):
    with open(fpkl, 'rb') as handle:
      peds = pickle.load(handle)
    for ped in peds:
      ped['old_fid20s'] = ped['fids20']
      ped['fids20'] = [i for i in range(ped['frame_start_20fps'], ped['frame_end_20fps']+1)]
    
    with open(fpkl, 'wb') as handle:
      pickle.dump(peds, handle)

  helper(os.path.join(out_ped_root, 'train.pkl'))
  helper(os.path.join(out_ped_root, 'test.pkl'))
      

def prepare_split_tmp():
  # using ANN_hanh1 only for feasibility test
  with open(os.path.join(out_ped_root, 'all_peds.pkl'), 'rb') as handle:
    peds = pickle.load(handle)

  import random
  ids = list(range(len(peds)))
  random.shuffle(ids)
  # 20% for testing
  test_ids = ids[:int(len(peds)*0.2)]

  # prepare the splits
  train = [peds[i] for i in range(len(peds)) if i not in test_ids]
  with open(os.path.join(out_ped_root, 'train_ANN_hanh1.pkl'), 'wb') as handle:
    pickle.dump(train, handle)
  test = [peds[i] for i in range(len(peds)) if i in test_ids]
  with open(os.path.join(out_ped_root, 'test_ANN_hanh1.pkl'), 'wb') as handle:
    pickle.dump(test, handle)
 

if __name__ == '__main__':
  if False:
    # test
    fsegm = '/sailhome/agalczak/crossing/datasets/JAAD_instance_segm/video_0131/00000001_segm.npy'
    parse_objs(fsegm)
    # fxml = '/sailhome/bingbin/crossing/datasets/JAAD_dataset/behavioral_data_xml/video_0001.xml'
    # fpos = '/vision2/u/caozj/datasets/JAAD_dataset/bounding_box_python/vbb_part/video_0001.npy'
    # parse_pedestrian(fxml, fpos)

  if False:
    prepare_data()

  if False:
    stip_segm2box_wrapper()

  if True:
    parse_pedestrian_wrapper()
    prepare_split_wrapper()
    # prepare_split_tmp()

  if True:
    fill_20fps()
