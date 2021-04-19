import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
import torch.utils.data as data
import torchvision.transforms as transforms

import time
import pdb

try:
  from .jaad import JAADDataset
  from .jaad_loc import JAADLocDataset
  from .stip import STIPDataset
except:
  from jaad import JAADDataset
  from jaad_loc import JAADLocDataset
  from stip import STIPDataset

def jaad_collate(batch):
  # each item in batch: a tuple of 
  # 1. ped_crops: (30, 224, 224, 3)
  # 2. masks: list of len 30: each = dict of ndarrays: (n_obj, 1080, 1920, 1)
  # 3. GT_act: binary ndarray: (30, 9)
  ped_crops = []
  masks = []
  GT_act, GT_bbox, GT_pose = [], [], []
  obj_bbox, obj_cls = [], []
  fids = []
  img_paths = []
  for each in batch:
    ped_crops += each['ped_crops'],
    masks += each['all_masks'],
    GT_act += each['GT_act'],
    GT_bbox += each['GT_bbox'],
    GT_pose += each['GT_pose'],
    obj_bbox += each['obj_bbox'],
    obj_cls += each['obj_cls'],
    obj_bbox_l += each['obj_bbox_l'],
    obj_cls_l += each['obj_cls_l'],
    obj_bbox_r += each['obj_bbox_r'],
    obj_cls_r += each['obj_cls_r'],
    fids += each['fids'],
    img_paths += each['img_paths'],
  ped_crops = torch.stack(ped_crops)
  GT_act = torch.stack(GT_act)
  GT_bbox = torch.stack(GT_bbox)
  GT_pose = torch.stack(GT_pose)
  fids = torch.stack(fids)
  ret = {
    'ped_crops': ped_crops,
    'all_masks': masks,
    'GT_act': GT_act,
    'GT_bbox': GT_bbox,
    'GT_pose': GT_pose,
    'obj_cls': obj_cls,
    'obj_bbox': obj_bbox,
    'fids': fids,
    'img_paths': img_paths,
  }
  if 'frames' in batch[0]:
    ret['frames'] = torch.stack([each['frames'] for each in batch], 0)
    ret['GT_driver_act'] = torch.stack([each['GT_driver_act'] for each in batch], 0)

  return ret


def jaad_loc_collate(batch):
  # each item in batch: a tuple of 
  # 1. ped_crops: (30, 224, 224, 3)
  # 2. masks: list of len 30: each = dict of ndarrays: (n_obj, 1080, 1920, 1)
  # 3. GT_act: binary ndarray: (30, 9)
  ped_crops = []
  masks = []
  GT_act, GT_ped_bbox = [], []
  obj_bbox, obj_cls = [], []
  fids = []
  for each in batch:
    ped_crops += each['ped_crops'],
    masks += each['all_masks'],
    GT_act += each['GT_act'],
    GT_ped_bbox += each['GT_ped_bbox'],
    obj_bbox += each['obj_bbox'],
    obj_cls += each['obj_cls'],
    fids += each['fids'],
  GT_act = torch.stack(GT_act)
  fids = torch.stack(fids)
  ret = {
    'ped_crops': ped_crops,
    'all_masks': masks,
    'GT_act': GT_act,
    'GT_ped_bbox': GT_ped_bbox,
    'obj_cls': obj_cls,
    'obj_bbox': obj_bbox,
    'fids': fids,
  }
  if 'frames' in batch[0]:
    ret['frames'] = torch.stack([each['frames'] for each in batch], 0)
  return ret


def stip_collate(batch):
  # each item in batch: a tuple of 
  # 1. ped_crops: (30, 224, 224, 3)
  # 2. masks: list of len 30: each = dict of ndarrays: (n_obj, 1080, 1920, 1)
  # 3. GT_act: binary ndarray: (30, 9)
  ped_crops = []
  masks = []
  GT_act, GT_bbox, GT_pose = [], [], []
  obj_bbox, obj_cls = [], []
  fids = []
  img_paths = []
  for each in batch:
    ped_crops += each['ped_crops'],
    masks += each['all_masks'],
    GT_act += each['GT_act'],
    GT_bbox += each['GT_bbox'],
    # GT_pose += each['GT_pose'],
    obj_bbox += each['obj_bbox'],
    obj_cls += each['obj_cls'],
    fids += each['fids'],
    img_paths += each['img_paths'],
  ped_crops = torch.stack(ped_crops)
  GT_act = torch.stack(GT_act)
  GT_bbox = torch.stack(GT_bbox)
  if len(GT_pose):
    GT_pose = torch.stack(GT_pose)
  fids = torch.stack(fids)
  ret = {
    'ped_crops': ped_crops,
    'all_masks': masks,
    'GT_act': GT_act,
    'GT_bbox': GT_bbox,
    'obj_cls': obj_cls,
    'obj_bbox': obj_bbox,
    'fids': fids,
    'img_paths': img_paths,
  }
  if len(GT_pose):
    ret['GT_pose'] = GT_pose
  if 'frames' in batch[0]:
    ret['frames'] = torch.stack([each['frames'] for each in batch], 0)
    ret['GT_driver_act'] = torch.stack([each['GT_driver_act'] for each in batch], 0)

  return ret



def get_data_loader(opt):
  if opt.dset_name.lower() == 'jaad':
    dset = JAADDataset(opt)
    print('Built JAADDataset.')
    collate_fn = jaad_collate

  elif opt.dset_name.lower() == 'jaad_loc':
    dset = JAADLocDataset(opt)
    print('Built JAADLocDataset.')
    collate_fn = jaad_loc_collate

  elif opt.dset_name.lower() == 'stip':
    dset = STIPDataset(opt)
    print('Built STIPDataset')
    collate_fn = stip_collate

  else:
    raise NotImplementedError('Sorry but we currently only support JAAD and STIP. ^ ^b')

  dloader = data.DataLoader(dset,
    batch_size=opt.batch_size,
    shuffle=opt.is_train,
    num_workers=opt.n_workers,
    pin_memory=True,
    collate_fn=collate_fn,
  )

  return dloader


def cache_all_objs(opt, cache_dir_name):
  opt.is_train = False

  opt.collapse_cls = 1
  cache_dir_root = '/sailhome/ajarno/STR-PIP/datasets/cache/'
  cache_dir = os.path.join(cache_dir_root, cache_dir_name)
  os.makedirs(cache_dir, exist_ok=True)
  opt.save_cache_format = os.path.join(cache_dir, opt.split, 'ped{}_fid{}.pkl')
  os.makedirs(os.path.dirname(opt.save_cache_format), exist_ok=True)

  dset = JAADDataset(opt)
  dloader = data.DataLoader(dset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=jaad_collate)

  t_start = time.time()
  for i,each in enumerate(dloader):
    if i%50 == 0 and i:
      print('{}: avg time: {:.3f}'.format(i, (time.time()-t_start) / 50))
      t_start = time.time()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dset-name', type=str, default='JAAD_loc')
  parser.add_argument('--annot-ped-format', type=str, default='/sailhome/ajarno/STR-PIP/datasets/annot_{}_ped.pkl')
  parser.add_argument('--annot-loc-format', type=str, default='/sailhome/ajarno/STR-PIP/datasets/annot_{}_loc.pkl')
  parser.add_argument('--is-train', type=int, default=1)
  parser.add_argument('--split', type=str, default='train')
  parser.add_argument('--seq-len', type=int, default=30)
  parser.add_argument('--ped-crop-size', type=tuple, default=(224, 224))
  parser.add_argument('--mask-size', type=tuple, default=(224, 224))
  parser.add_argument('--collapse-cls', type=int, default=0,
                      help='Whether to merge the classes. If 1 then each item in masks is a dict keyed by cls, otherwise a list.')
  parser.add_argument('--img-path-format', type=str,
                      default='/sailhome/ajarno/STR-PIP/datasets/JAAD_dataset/JAAD_clip_images/video_{:04d}.mp4/{:d}.jpg')
  parser.add_argument('--fsegm-format', type=str,
                      default='/sailhome/ajarno/STR-PIP/datasets/JAAD_instance_segm/video_{:04d}/{:08d}_segm.npy')
  parser.add_argument('--save-cache-format', type=str, default='')
  parser.add_argument('--cache-format', type=str, default='')
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--n-workers', type=int, default=0)
  
  # added to test loader
  parser.add_argument('--rand-test', type=int, default=1)
  parser.add_argument('--predict', type=int, default=0)
  parser.add_argument('--predict-k', type=int, default=0)
  parser.add_argument('--combine-method', type=str, default='none')
  parser.add_argument('--load-cache', type=str, default='masks')
  parser.add_argument('--cache-obj-bbox-format', type=str,
                             default='/sailhome/ajarno/STR-PIP/datasets/cache/obj_bbox_merged/vid{:08d}.pkl')

  opt = parser.parse_args()
  opt.save_cache_format = '/sailhome/ajarno/STR-PIP/datasets/cache/jaad_loc/{}/vid{}_fid{}.pkl'
  opt.cache_format = opt.save_cache_format
  opt.seq_len = 1
  opt.split = 'test'

  if True:
    # test dloader
    dloader = get_data_loader(opt)
    for i,vid in enumerate(dloader.dataset.vids):
      print('vid:', vid)
      annot = dloader.dataset.annots[vid]
      n_frames = len(annot['act'])
      for fid in range(n_frames):
        fcache = opt.cache_format.format(opt.split, vid, fid+1)
        if os.path.exists(fcache):
          continue
        dloader.dataset.__getitem__(i, fid_start=fid)
