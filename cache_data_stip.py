# NOTE: this script uses segmentation files only
# and does not rely on pedestrian annotations.

# python imports
import os
import sys
from glob import glob
import pickle
import time
import numpy as np


# local imports
import data
import utils
from utils.data_proc_stip import parse_objs

import pdb


# def cache_masks():
#   opt, logger = utils.build(is_train=False)
#   opt.combine_method = ''
#   opt.split = 'train'
#   cache_dir_name = 'jaad_collapse{}'.format('_'+opt.combine_method if opt.combine_method else '')
#   data.cache_all_objs(opt, cache_dir_name)
# 
# 
# def cache_crops():
#   fnpy_root = '/sailhome/bingbin/STR-PIP/datasets/JAAD_instance_segm'
#   fpkl_root = '/sailhome/bingbin/STR-PIP/datasets/cache/JAAD_instance_crops'
#   utils.get_obj_crops(fnpy_root, fpkl_root)

def add_obj_bbox(view=''):
  if not view:
    vid_root = '/vision/group/prolix/instances_20fps/stip_instances/'
    fobj_root = '/vision/group/prolix/processed/obj_bbox_20fps'
  elif view == 'center':
    vid_root = '/vision/group/prolix/instances_20fps/stip_instances/'
    fobj_root = '/vision/group/prolix/processed/center/obj_bbox_20fps'
  else: # view = 'left' or 'right'
    vid_root = '/vision/group/prolix/stip_side/{}/instances_20fps/'.format(view)
    fobj_root = '/vision/group/prolix/processed/{}/obj_bbox_20fps/'.format(view)
  os.makedirs(fobj_root, exist_ok=True)

  def helper(vid_range, split):
    for dir_vid in vid_range:
      print(dir_vid)
      sys.stdout.flush()
      t_start = time.time()

      if not view or view == 'center':
        segs = sorted(glob(os.path.join(vid_root, dir_vid, 'inference', '*--*')))
      else:
        segs = sorted(glob(os.path.join(vid_root, dir_vid, '*--*')))

      for seg in segs:
        if not view or view == 'center':
          fsegms = sorted(glob(os.path.join(seg, '*_segm.npy')))
        else:
          fsegms = sorted(glob(os.path.join(seg, '*.pkl')))
        for i, fsegm in enumerate(fsegms):
          if i and i%100 == 0:
            print('Time per frame:', (time.time() - t_start)/i)
            sys.stdout.flush()
          fid = os.path.basename(fsegm).split('_')[0]
          fbbox = os.path.join(fobj_root, '{:s}_seg{:s}_fid{:s}.pkl'.format(dir_vid, os.path.basename(seg), fid))
          # if 'ANN_conor1_seg12:22--12:59_fid0000015483.pkl' not in fbbox:
          #   continue
          if os.path.exists(fbbox):
            continue
          if not os.path.exists(fsegm):
            print('File does not exist:', fsegm)
            continue
          objs = parse_objs(fsegm)
          dobjs = {cls:[] for cls in range(1,5)}
          for cls, masks in objs.items():
            for mask in masks:
              try:
                if len(mask.shape) == 3:
                  h, w, c = mask.shape
                  if c != 1:
                    raise ValueError('Each mask should have shape (1080, 1920, 1)')
                  mask = mask.reshape(h, w)
                x_pos = mask.sum(0).nonzero()[0]
                if not len(x_pos):
                  x_pos = [0,0]
                x_min, x_max = x_pos[0], x_pos[-1]
                y_pos = mask.sum(1).nonzero()[0]
                if not len(y_pos):
                  y_pos = [0,0]
                y_min, y_max = y_pos[0], y_pos[-1]
                # bbox: [x_min, y_min, w, h]; same as bbox for ped['pos_GT']
                bbox = [x_min, y_min, x_max-x_min, y_max-y_min]
              except Exception as e:
                print(e)
                pdb.set_trace()
 
              dobjs[cls] += bbox,
          with open(fbbox, 'wb') as handle:
            pickle.dump(dobjs, handle)

  vids = sorted(glob(os.path.join(vid_root, '*_*')))
  vids = [os.path.basename(vid) for vid in vids]
  vids_test = ['downtown_ann_3-09-28-2017', 'downtown_palo_alto_6', 'dt_san_jose_4', 'mountain_view_4', 
  'sf_soma_2']
  vids_train = [vid for vid in vids if vid not in vids_test]

  # tmp
  # vids_train = ['downtown_ann_1-09-27-2017', 'downtown_ann_2-09-27-2017', 'downtown_ann_3-09-27-2017', 'downtown_ann_1-09-28-2017']
  # vids_test = []

  if True:
    helper(vids_train, 'train')
  if False:
    helper(vids_test, 'test')

def merge_and_flat(vrange, view=''):
  """
  Merge fids in a vid and flatten the classes
  """
  if not view:
    pkl_in_root = '/vision/group/prolix/processed/obj_bbox_20fps'
    pkl_out_root = '/vision/group/prolix/processed/obj_bbox_20fps_merged'
  else:
    pkl_in_root = '/vision/group/prolix/processed/{}/obj_bbox_20fps/'.format(view)
    pkl_out_root = '/vision/group/prolix/processed/{}/obj_bbox_20fps_merged'.format(view)
  os.makedirs(pkl_out_root, exist_ok=True)

  for vid in vrange:
    print(vid)
    fpkls = sorted(glob(os.path.join(pkl_in_root, '{:s}*pkl'.format(vid))))
    segs = list(set([fpkl.split('_fid')[0] for fpkl in fpkls]))
    print('# segs:', len(segs))
    for seg in segs:
      fpkls = sorted(glob(seg+'*pkl'))
      print(vid, len(fpkls))
      sys.stdout.flush()
      # merged = [[] for _ in range(len(fpkls))]
      merged_bbox = []
      merged_cls = []
      t_start = time.time()
      for fpkl in fpkls:
        try:
          with open(fpkl, 'rb') as handle:
            data = pickle.load(handle)
        except:
          pdb.set_trace()
        curr_bbox = []
        cls = []
        for c in [1,2,3,4]:
          for bbox in data[c]:
            cls += c,
            curr_bbox += bbox,
        merged_bbox += np.array(curr_bbox),
        merged_cls += np.array(cls),
        
      seg = seg.split('seg')[-1]
      fpkl_out = os.path.join(pkl_out_root, '{}_seg{}.pkl'.format(vid, seg))
      with open(fpkl_out, 'wb') as handle:
        dout = {
          'obj_cls': merged_cls,
          'obj_bbox': merged_bbox,
        }
        pickle.dump(dout, handle)
      print('avg time: ', (time.time()-t_start) / len(fpkls))


def merge_and_flat_wrapper(view=''):
  vid_root = '/vision/group/prolix/instances_20fps/stip_instances/'
  vids = sorted(glob(os.path.join(vid_root, '*_*')))
  vids = [os.path.basename(vid) for vid in vids]
  vids_test = ['downtown_ann_3-09-28-2017', 'downtown_palo_alto_6', 'dt_san_jose_4', 'mountain_view_4', 
  'sf_soma_2']
  vids_train = [vid for vid in vids if vid not in vids_test]

  # tmp
  # vids_train = ['downtown_ann_1-09-27-2017', 'downtown_ann_2-09-27-2017', 'downtown_ann_3-09-27-2017', 'downtown_ann_1-09-28-2017']
  # vids_test = []

  if True:
    merge_and_flat(vids_train, view=view)
  if False:
    merge_and_flat(vids_test, view=view)


def cache_loc():
  def helper(annots):
    for vid in annots:
      annot = annots[vid]
      n_frames = len(annot['act'])
      for fid in range(n_frames):
        # fid in ped cache file name: 1 based
        loc
        # ped:
        # ped['ped_crops']: ndarray: (3, 224, 224)
        # ped['masks']: tensor: [n_objs, 224, 224]


if __name__ == '__main__':
  # cache_masks()
  # cache_crops()

  # add_obj_bbox(view='left')
  # merge_and_flat_wrapper(view='left')
  # add_obj_bbox(view='right')
  # merge_and_flat_wrapper(view='right')
  # add_obj_bbox(view='center')
  merge_and_flat_wrapper(view='center')

  # merge_and_flat(range(1, 200))
  # merge_and_flat(range(100, 200))
  # merge_and_flat(range(200, 300))
  # merge_and_flat(range(100, 347))
  # merge_and_flat(range(200, 347))

