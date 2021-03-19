import os
from glob import glob
import numpy as np
import pickle
import cv2
import xml.etree.ElementTree as ET
from pycocotools import mask as maskUtils

import pdb

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
  segms = np.load(fnpy, allow_pickle=True)

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


pedestrian_act_map = {
  'clear path': 0,
  'crossing': 1,
  'handwave': 2,
  'looking': 3,
  'nod': 4,
  'slow down': 5,
  'speed up': 6,
  'standing': 7,
  'walking': 8,
}

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

def parse_pedestrian(fxml, fpos_GT, fpos_pred='', fpose=''):
  """
  parse xml (action label)
  """
  e = ET.parse(fxml).getroot()
  # peds: dict of id to a list of per-frame acts.
  # Action labels at each frame is a one-hot vec of length 9 (i.e. len(pedestrian_act_map)).
  peds = {}
  nframes = int(e.get('num_frames'))
  for child in e.getchildren():
    if child.tag != 'actions':
      continue

    for each in child.getchildren():
      if 'pedestrian' not in each.tag: # e.g. Driver
        continue

      # NOTE: `pid` starts at 1.
      tag = each.tag
      if tag == 'pedestrian':
        pid = 1
        tag = 'pedestrian1'
      else:
        pid = int(tag[len('pedestrian'):])
      # NOTE: change indexing from 'pid' to 'each.tag'
      peds[tag] = {'act': [[0] * len(pedestrian_act_map) for _ in range(nframes)]}
      peds[tag]['tag'] = tag
      peds[tag]['pid'] = pid
      pacts = each.getchildren()
      for act in pacts:
        act_cls = pedestrian_act_map[act.get('id').lower()]
        for t in range(int(act.get('start_frame'))-1, int(act.get('end_frame'))):
          peds[tag]['act'][t][act_cls] = 1

  if fpose:
    """
    parse pose
    """
    pose_data = np.load(fpose, encoding='latin1', allow_pickle=True).item()
    for ped_tag, frames in sorted(pose_data.items()):
      if ped_tag not in peds:
        continue
      peds[ped_tag]['pose'] = [[] for _ in range(nframes)]
      peds[ped_tag]['pos_Pose'] = [[] for _ in range(nframes)]
  
      # NOTE fid are 0-indexed in pose_data
      for frame in sorted(frames):
        fid = frame[0]
        peds[ped_tag]['pose'][fid] = np.array(frame[2][0])[joint_ids]
        peds[ped_tag]['pos_Pose'][fid] = frame[1]['pos']

  """
  parse position
  """
  poccl = {ped_tag:[[] for _ in range(nframes)] for ped_tag in peds}

  # NOTE: not all pedestrians in the npy files have labels.
  # We only use those with action labels in xml.
  data = np.load(fpos_GT, allow_pickle=True).item()
  assert(data['nFrame'] == nframes), print("nFrame mismatch: xml:{} / npy: {}".format(nframes, data['nFrame']))
  
  assert(len(data['objLists']) == nframes), print("nFrame mismatch: xml:{} / npy-->objLists: {}".format(nframes, len(data['objLists'])))

  ppos_GT = {ped_tag:[[] for _ in range(nframes)] for ped_tag in peds}
  # pdb.set_trace()
  for fid,frame in enumerate(data['objLists']):
    # frame: dict w/ keys: 'id', 'pos', 'posv', 'occl', 'lock'
    for ped in frame:
      pid = ped['id'][0]
      for ped_tag in peds:
        if peds[ped_tag]['pid'] == pid:
          ppos_GT[ped_tag][fid] = ped['pos'] # (x, y, w, h)
          poccl[ped_tag][fid] = ped['occl'][0]
          break
      # if ped_tag in peds:
      #   ppos_GT[ped_tag][fid] = ped['pos'] # (x, y, w, h)
      #   poccl[ped_tag][fid] = ped['occl'][0]

  for ped_tag in peds:
    pid = peds[ped_tag]['pid']
    peds[ped_tag]['frame_start'] = data['objStr'][pid-1]-1
    peds[ped_tag]['frame_end'] = data['objEnd'][pid-1]-1
    peds[ped_tag]['pos_GT'] = ppos_GT[ped_tag]
    peds[ped_tag]['occl'] = poccl[ped_tag]

  if fpos_pred:
    raise NotImplementedError('cannot parse pedestrian segm for now. Sorry!!')

  return peds


def prepare_data():
  # object GT directory
  obj_root = '/sailhome/ajarno/STR-PIP/datasets/JAAD_instance_segm'
  fobj_dir_format = os.path.join(obj_root, 'video_{:04d}')
  # pedestrian GT files
  #ped_root = '/sailhome/ajarno/STR-PIP/datasets/JAAD_dataset/'
  ped_root = '/vision/u/caozj1995/data/JAAD_dataset/'
  fxml_format = os.path.join(ped_root, 'behavioral_data_xml', 'video_{:04d}.xml')
  fpose_format = os.path.join('/vision2/u/mangalam/JAAD/openpose_track_with_pose/', 'video_{:04d}.npy')
  fpos_GT_format = os.path.join(ped_root, 'bounding_box_python', 'vbb_part', 'video_{:04d}.npy')

  def prepare_split(vid_range):
    
    all_peds = []
    for vid in vid_range:
      # print(vid)

      if True:
        # objects
        fobj_dir = fobj_dir_format.format(vid)
        fsegms = sorted(glob(os.path.join(fobj_dir, '*_segm.npy')))
        pdb.set_trace()
  
        frame_objs = []
        for fid,fsegm in enumerate(fsegms):
          print('fid:', fid)
          frame_objs += parse_objs(fsegm),

        print('Finished objectS')
  
      if True:
        # pedestrians
        fxml = fxml_format.format(vid)
        fpose = fpose_format.format(vid)
        fpose = ''
        fpos_GT = fpos_GT_format.format(vid)
        ped_label = parse_pedestrian(fxml, fpos_GT, fpose=fpose)
        if len(ped_label) == 0:
          print('No pedestrian: vid:', vid)
  
        for ped in ped_label.values():
          ped['vid'] = vid
          all_peds += ped,

    return all_peds


  annot_root = '/sailhome/ajarno/STR-PIP/datasets/'
  # Train
  print('Processing training data...')
  train_range = range(1, 250+1)
  annot_train = prepare_split(train_range)
  print('# Train:', len(annot_train))
  with open(os.path.join(annot_root, 'annot_train_ped_withTag_sanityNoPose.pkl'), 'wb') as handle:
    pickle.dump(annot_train, handle)
  print('Training data ready.\n')

  # Test
  print('Processing testing data...')
  test_range = range(251, 346+1)
  annot_test = prepare_split(test_range)
  print('# Test:', len(annot_test))
  with open(os.path.join(annot_root, 'annot_test_ped_withTag_sanityNoPose.pkl'), 'wb') as handle:
    pickle.dump(annot_test, handle)
  print('Testing data ready.\n')

 

if __name__ == '__main__':
  if False:
    # test
    fsegm = '/sailhome/bingbin/STR-PIP/datasets/JAAD_instance_segm/video_0131/00000001_segm.npy'
    parse_objs(fsegm)
    # fxml = '/sailhome/bingbin/STR-PIP/datasets/JAAD_dataset/behavioral_data_xml/video_0001.xml'
    # fpos = '/vision2/u/caozj/datasets/JAAD_dataset/bounding_box_python/vbb_part/video_0001.npy'
    # parse_pedestrian(fxml, fpos)

  if True:
    prepare_data()

  if False:
    stip_segm2box_wrapper()
