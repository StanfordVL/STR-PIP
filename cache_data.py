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
from utils.data_proc import parse_objs

import pdb


def cache_masks():
  opt, logger = utils.build(is_train=False)
  opt.combine_method = ''
  opt.split = 'train'
  cache_dir_name = 'jaad_collapse{}'.format('_'+opt.combine_method if opt.combine_method else '')
  data.cache_all_objs(opt, cache_dir_name)


def cache_crops():
  fnpy_root = '/sailhome/bingbin/STR-PIP/datasets/JAAD_instance_segm'
  fpkl_root = '/sailhome/bingbin/STR-PIP/datasets/cache/JAAD_instance_crops'
  utils.get_obj_crops(fnpy_root, fpkl_root)

def add_obj_bbox():
  fnpy_root = '/sailhome/bingbin/STR-PIP/datasets/JAAD_instance_segm'
  # fpkl_root = '/sailhome/bingbin/STR-PIP/datasets/cache/jaad_collapse'
  fobj_root = '/sailhome/bingbin/STR-PIP/datasets/cache/obj_bbox'
  os.makedirs(fobj_root, exist_ok=True)
  dir_vids = sorted(glob(os.path.join(fnpy_root, 'vid*')))

  def helper(vid_range, split):
    for dir_vid in vid_range:
      print(dir_vid)
      sys.stdout.flush()
      vid = int(os.path.basename(dir_vid).split('_')[1])
      t_start = time.time()
      fsegms = sorted(glob(os.path.join(dir_vid, '*_segm.npy')))
      for i, fsegm in enumerate(fsegms):
        if i and i%100 == 0:
          print('Time per frame:', (time.time() - t_start)/i)
          sys.stdout.flush()
        # Note: 'fid' is 0-based for segm, but 1-based for images and caches.
        fid = os.path.basename(fsegm).split('_')[0]
        fbbox = os.path.join(fobj_root, 'vid{:08d}_fid{:s}.pkl'.format(vid, fid))
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

  if False:
    vids_train = dir_vids[:250]
    helper(vids_train, 'train')
  if True:
    vids_test = dir_vids[250:]
    helper(vids_test, 'test')

def merge_and_flat(vrange):
  """
  Merge fids in a vid and flatten the classes
  """
  pkl_in_root = '/sailhome/bingbin/STR-PIP/datasets/cache/obj_bbox'
  pkl_out_root = '/sailhome/bingbin/STR-PIP/datasets/cache/obj_bbox_merged'
  os.makedirs(pkl_out_root, exist_ok=True)
  # for vid in range(1, 347):
  for vid in vrange:
    fpkls = sorted(glob(os.path.join(pkl_in_root, 'vid{:08d}*pkl'.format(vid))))
    print(vid, len(fpkls))
    sys.stdout.flush()
    # merged = [[] for _ in range(len(fpkls))]
    merged_bbox = []
    merged_cls = []
    t_start = time.time()
    for fpkl in fpkls:
      with open(fpkl, 'rb') as handle:
        data = pickle.load(handle)
      curr_bbox = []
      cls = []
      for c in [1,2,3,4]:
        for bbox in data[c]:
          cls += c,
          curr_bbox += bbox,
      merged_bbox += np.array(curr_bbox),
      merged_cls += np.array(cls),
      
    fpkl_out = os.path.join(pkl_out_root, 'vid{:08d}.pkl'.format(vid))
    with open(fpkl_out, 'wb') as handle:
      dout = {
        'obj_cls': merged_cls,
        'obj_bbox': merged_bbox,
      }
      pickle.dump(dout, handle)
    print('avg time: ', (time.time()-t_start) / len(fpkls))

if __name__ == '__main__':
  # cache_masks()
  cache_crops()
  add_obj_bbox()

  # merge_and_flat(range(1, 200))
  # merge_and_flat(range(100, 200))
  # merge_and_flat(range(200, 300))
  # merge_and_flat(range(100, 347))
  merge_and_flat(range(1, 347))

