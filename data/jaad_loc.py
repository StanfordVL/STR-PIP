import os
from glob import glob
import numpy as np
import pickle
import cv2
import random
random.seed(2019)
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import pdb
import time

from utils.data_proc import parse_objs

class JAADLocDataset(data.Dataset):
  def __init__(self, opt):
    # annot_ped_format, is_train, split,
    # seq_len, ped_crop_size, mask_size, collapse_cls,
    # img_path_format, fsegm_format):

    self.split = opt.split
    annot_loc = opt.annot_loc_format.format(self.split)
    with open(annot_loc, 'rb') as handle:
      self.annots = pickle.load(handle)
    self.vids = sorted(self.annots.keys())

    self.is_train = opt.is_train
    self.rand_test = opt.rand_test
    self.seq_len = opt.seq_len
    self.predict = opt.predict
    if self.predict:
      self.all_seq_len = self.seq_len + opt.pred_seq_len
    else:
      self.all_seq_len = self.seq_len
    self.predict_k = opt.predict_k
    if self.predict_k:
      self.all_seq_len += self.predict_k
    self.ped_crop_size = opt.ped_crop_size
    self.mask_size = opt.mask_size
    self.collapse_cls = opt.collapse_cls
    self.combine_method = opt.combine_method

    self.img_path_format = opt.img_path_format
    self.fsegm_format = opt.fsegm_format
    self.save_cache_format = opt.save_cache_format
    self.load_cache = opt.load_cache
    self.cache_format = opt.cache_format
    self.cache_obj_bbox_format = opt.cache_obj_bbox_format

  def __len__(self):
    return len(self.vids)

  def __getitem__(self, idx, fid_start=-1):
    t_start = time.time()

    vid = self.vids[idx]
    annot = self.annots[vid]

    # print('vid:', vid)

    if fid_start != -1:
      f_start = fid_start
    elif self.is_train or self.rand_test:
      # train: randomly sample all_seq_len number of frames
      f_start = random.randint(0, len(annot['act'])-self.all_seq_len+1)
    elif fid_start == -1:
      f_start = 0

    if self.is_train or self.rand_test or fid_start != -1:
      if self.predict_k:
        fids = [min(len(annot['act'])-1, f_start+i) for i in range(self.seq_len)]
        fids += min(len(annot['act'])-1, f_start+self.seq_len-1+self.predict_k),
      else:
        fids = [min(len(annot['act'])-1, f_start+i) for i in range(self.all_seq_len)]
    else:
      # take the entire video
      fids = range(0, len(annot['act']))

    GT_act = [annot['act'][fid] for fid in fids]
    GT_act = np.stack(GT_act)
    GT_act = torch.tensor(GT_act)

    GT_ped_bbox = [[pos.astype(np.float) for pos in annot['ped_pos'][fid]] for fid in fids]

    with open(self.cache_obj_bbox_format.format(vid), 'rb') as handle:
      data = pickle.load(handle)
    obj_cls = [data['obj_cls'][fid] for fid in fids]
    obj_bbox = [data['obj_bbox'][fid] for fid in fids]

    frames = []
    for fid in fids[:self.seq_len]:
      img_path = self.img_path_format.format(vid, fid+1) # +1 since fid is 0-based while the frame path starts at 1.
      img = cv2.imread(img_path)
      img = cv2.resize(img, (224,224)).transpose((2,0,1))
      frames += img,

    ret = {
      'GT_act': GT_act,
      'GT_ped_bbox': GT_ped_bbox,
      'obj_cls': obj_cls,
      'obj_bbox': obj_bbox,
      'frames': torch.tensor(frames),
      'fids': torch.tensor(np.array(fids)),
    }
    ped_crops = []
    all_masks = []

    # only the first seq_len fids are input data
    fids = fids[:self.seq_len]

    if self.load_cache == 'masks':
      for i,fid in enumerate(fids):
        # NOTE: fid for masks is 1-based.
        fcache = self.cache_format.format(self.split, vid, fid+1)
        # print(fcache)
        if os.path.exists(fcache):
          with open(fcache, 'rb') as handle:
            data = pickle.load(handle)
            ped_crops += data['ped_crops'],
            all_masks += data['masks'],
            
            if 'max' not in fcache:
              # 1 mask per obj: check for # objs
              if type(data['masks']) == dict:
                n_objs = len([each for val in data['masks'].values() for each in val])
              else:
                n_objs = len(data['masks'])
              if n_objs != len(obj_bbox[i]):
                print('JAAD: n_objs mismatch')
                pdb.set_trace()
        else:
          try:
            ped_crop, cls_masks = self.get_vid_fid(vid, annot['ped_pos'][fid], fid)
          except Exception as e:
            print(e)
            pdb.set_trace()
          ped_crops += ped_crop,
          all_masks += cls_masks,
          if type(cls_masks) == dict:
            n_objs = len([each for val in cls_masks.values() for each in val])
          else:
            n_objs = len(cls_masks)
          if n_objs != len(obj_bbox[i]):
            print('JAAD: n_objs mismatch')
            pdb.set_trace()

      ret['ped_crops'] = ped_crops
      ret['all_masks'] = all_masks
      
      # pdb.set_trace()
      n_peds = sum([len(each) for each in ped_crops])
      n_objs = sum([len(each) for each in all_masks])
      # print('n_peds:{} / n_objs:{}'.format(n_peds, n_objs))
      return ret

    elif self.load_cache == 'feats':
      ped_feats = []
      ctxt_feats = []
      for fid in fids:
        # NOTE: fid for feats is 0-based.
        # with open(self.cache_format.format(self.split, vid, fid), 'rb') as handle:
        with open(self.cache_format.format(self.split, idx, fid), 'rb') as handle:
          data = pickle.load(handle)
          ped = data['ped_feats'] # shape: 1, 512
          ctxt = data['ctxt_feats'] # shape: n_objs, 512

          ped_feats += ped,
          ctxt_feats += ctxt,
      # ped_feats = torch.stack(ped_feats, 0)

      ret['ped_crops'] = ped_feats
      ret['all_masks'] = ctxt_feats
      return ret

    elif self.load_cache == 'pos':
      ret['ped_crops'] = torch.zeros([1,1,512])
      ret['all_masks'] = torch.zeros([1,1,512])
      return ret

    for fid in fids:
      ped_crop, cls_masks = self.get_ped_fid(ped, fid, idx)
      ped_crops += ped_crop,
      all_masks += cls_masks,

    # shape: [n_frames, self.ped_crop_size[0], self.ped_crop_size[1]]

    ped_crops = np.stack(ped_crops)
    ped_crops = torch.Tensor(ped_crops)

    # print('time per item:', time.time()-t_start)
    ret['ped_crops'] = ped_crops
    ret['all_masks'] = all_masks
    return ret


  def get_vid_fid(self, vid, peds, fid):
      """
      Prepare ped_crop and obj masks for given ped and fid.
      """

      ped_crops = []
      if len(peds) == 0:
        ped_crops = [np.zeros([3, 224, 224])]
        # if no bbox, take the entire frame
        x,y,w,h = 0,0,-1,-1
      else:
        img_path = self.img_path_format.format(vid, fid+1) # +1 since fid is 0-based while the frame path starts at 1.
        img = cv2.imread(img_path)

        # pedestrian crops
        for ped in peds:
          x, y, w, h = ped
          x, y, w, h = int(x), int(y), int(w), int(h)
  
          try:
            ped_crop = img[y:y+h, x:x+w]
          except Exception as e:
            print(e)
            print('img_path:', img_path)
            print('x:{}, y:{}, w:{}, h:{}'.format(x, y, w, h))
          ped_crop = cv2.resize(ped_crop, self.ped_crop_size)
          ped_crop = ped_crop.transpose((2,0,1))
          ped_crops += ped_crop,
      
      # obj masks
      fsegm = self.fsegm_format.format(vid, fid)
      objs = parse_objs(fsegm)
      if self.collapse_cls:
        cls_masks = []
      else:
        cls_masks = {}
      for cls, masks in objs.items():
        if not self.collapse_cls:
          cls_masks[cls] = []
        for mask in masks:
          mask[y:y+h, x:x+w] = 1
          if self.combine_method == 'pair':
            # crop out the union bbox of ped + obj
            # note that didn't check empty.
            x_pos = mask.sum(0).nonzero()[0]
            x_min, x_max = x_pos[0], x_pos[-1]
            y_pos = mask.sum(1).nonzero()[1]
            y_min, y_max = y_pos[0], y_pos[-1]
            mask = mask[y_min:y_max+1, x_min:x_max+1]
          mask = cv2.resize(mask, self.mask_size)
          mask = torch.tensor(mask)
          # mask = torch.stack([mask, mask, mask])
          if self.collapse_cls:
            cls_masks += mask,
          else:
            # TODO: transform the mask: e.g. crop & norm over the union
            cls_masks[cls] += mask,
        if not self.collapse_cls:
          cls_masks[cls] = torch.stack(cls_masks[cls])
          if self.combine_method == 'sum':
            cls_masks[cls] = cls_masks[cls].sum(0)
          elif self.combine_method == 'max':
            cls_masks[cls], _ = cls_masks[cls].max(0)

      if self.collapse_cls:
        if len(cls_masks) != 0:
          cls_masks = torch.stack(cls_masks)
          if self.combine_method == 'sum':
            cls_masks = cls_masks.sum(0)
          elif self.combine_method == 'max':
            cls_masks, _ = cls_masks.max(0)
        else:
          # no objects in the frame
          if self.combine_method:
            # e.g. 'sum' or 'max'
            cls_masks = torch.zeros(self.mask_size)
          else:
            cls_masks = torch.zeros([0, self.mask_size[0], self.mask_size[1]])

      if self.cache_format:
        with open(self.cache_format.format(self.split, vid, fid+1), 'wb') as handle:
          cache = {
            'ped_crops': ped_crops,
            'masks': cls_masks.data if self.collapse_cls else {cls:cls_masks[cls].data for cls in cls_masks},
          }
          pickle.dump(cache, handle)

      return ped_crops, cls_masks


