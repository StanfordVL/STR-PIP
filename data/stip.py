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

class STIPDataset(data.Dataset):
  def __init__(self, opt):
    # annot_ped_format, is_train, split,
    # seq_len, ped_crop_size, mask_size, collapse_cls,
    # img_path_format, fsegm_format):

    self.split = opt.split
    annot_ped = opt.annot_ped_format.format(self.split)
    with open(annot_ped, 'rb') as handle:
      self.peds = pickle.load(handle)

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

    # TODO: tmp img path
    # format args: video (vid), segment (seg), frame id (fid)
    self.driver_act_format = opt.driver_act_format
    self.fsegm_format = opt.fsegm_format
    # TODO: tmp semgation file path
    # format args: video (vid), segment (seg), frame id (fid)
    self.view = opt.view
    if self.view == 'center':
      self.img_path_format = '/vision/group/prolix/images_20fps/{:s}/{:s}/{:010d}.png'
      self.fsegm_format = '/vision/group/prolix/instances_20fps/stip_instances/{:s}/inference/{:s}/{:010d}_segm.npy'
    elif self.view == 'all':
      # NOTE: this setting should use the cached feats directly, rather than segm files.
      self.img_path_format = '/vision/group/prolix/images_20fps/{:s}/{:s}/{:010d}.png'
      self.fsegm_format = '' # TODO
    elif self.view == 'left':
      self.img_path_format = '/vision/group/prolix/stip_side/left/images_20fps/{:s}/{:s}/{:010d}.png'
      self.fsegm_format = '/vision/group/prolix/stip_side/left/instances_20fps/{:s}/{:s}/{:010d}.pkl'
    elif self.view == 'right':
      self.img_path_format = '/vision/group/prolix/stip_side/left/images_20fps/{:s}/{:s}/{:010d}.png'
      self.fsegm_format = '/vision/group/prolix/stip_side/right/instances_20fps/{:s}/{:s}/{:010d}.pkl'

    self.cache_format = opt.cache_format
    self.cache_obj_bbox_format = opt.cache_obj_bbox_format
    self.save_cache_format = opt.save_cache_format
    self.load_cache = opt.load_cache

    self.use_driver = opt.use_driver
    self.use_pose = opt.use_pose

    self.idx_map = {i:i for i in range(len(self.peds))}
    if self.load_cache == 'feats':
      # keep only valid peds
      self.invalid_pid = []
      for idx in range(len(self.peds)):
        self.__getitem__(idx)

      new_idx_map = {}
      idx = 0
      for pid in range(len(self.peds)):
        if pid in self.invalid_pid:
          continue
        new_idx_map[idx] = pid
        idx += 1
      self.idx_map = new_idx_map

  def __len__(self):
    return len(self.idx_map)

  def __getitem__(self, idx, fid_start=-1):
    # idx = self.keys[idx]
    
    # if self.split == 'test' and self.view == 'all' and idx == 0:
    #   return None

    t_start = time.time()

    pid = self.idx_map[idx]
    ped = self.peds[pid]
    if self.use_pose and 'pose' not in ped:
      idx = random.randint(1, self.__len__()-1) # TODO
      return self.__getitem__(idx, fid_start)

    # starting index relative to ped['fid20s']
    fid_start_rel = None
    if fid_start != -1:
      f_start_rel = fid_start
    elif (self.is_train or self.rand_test) and len(ped['fids20'])>self.all_seq_len:
      # train: randomly sample all_seq_len number of frames
      f_start_rel = random.randrange(len(ped['fids20'])-self.all_seq_len+1)
    elif fid_start == -1:
      f_start_rel = 0

    # indices relative to ped['fid20s']
    fids_rel = None
    ped['fids20'] = ped['old_fid20s'] # TODO: tmp fix
    ped['fids20'] = sorted(ped['fids20']) # TODO: why are the fids not sorted originally?
    if self.is_train or self.rand_test or fid_start != -1:
      if self.predict_k:
        fids_rel = [min(len(ped['fids20'])-1, f_start_rel+i) for i in range(self.seq_len)]
      else:
        fids_rel = [min(len(ped['fids20'])-1, f_start_rel+i) for i in range(self.all_seq_len)]
    else:
      fids_rel = list(range(len(ped['fids20'])))

    GT_act = [ped['act'][fid] for fid in fids_rel]
    GT_act = np.stack(GT_act)
    GT_act = torch.Tensor(GT_act)
    GT_act = GT_act.unsqueeze(-1)

    GT_bbox = [ped['pos_GT'][fid].astype(np.float) if len(ped['pos_GT'][fid]) else np.zeros(4) for fid in fids_rel]
    GT_bbox = np.stack(GT_bbox)
    GT_bbox = torch.Tensor(GT_bbox)

    vid = ped['vid']

    if self.use_pose:
      GT_pose = [ped['pose'][fid] for fid in fids_rel]
      n_zero = 0
      for i,each in enumerate(GT_pose):
        if len(each) == 0:
          # pad with 0-speed pose
          GT_pose[i] = np.zeros([9, 3]) if i==0 else GT_pose[i-1].copy()

          # pad with 0s
          # GT_pose[i] = np.zeros([9, 3])
          n_zero += 1
      GT_pose = np.stack(GT_pose)
      GT_pose = torch.Tensor(GT_pose)
    else:
      GT_pose = torch.Tensor([0])

    # TODO: merge the bboxes

    # fbbox = self.cache_obj_bbox_format.format(vid, ped['seg'])
    # with open(fbbox, 'rb') as handle:
    #   bbox_data = pickle.load(handle)
    #   # obj_cls += data['obj_cls'],
    #   # obj_bbox += data['obj_bbox'],

    # cache_map = {'c': self.cache_obj_bbox_format}
    # if self.cache_obj_bbox_format_left:
    #   cache_map['l'] = self.cache_obj_bbox_format_left
    # if self.cache_obj_bbox_format_right:
    #   cache_map['r'] = self.cache_obj_bbox_format_right

    obj_cls, obj_bbox = [], []
    fbbox = self.cache_obj_bbox_format.format(vid, ped['seg'])
    if not os.path.exists(fbbox):
      print('Path does not exist:{}. \n Skipping...'.format(fbbox))
      return None

    # pdb.set_trace()

    with open(fbbox, 'rb') as handle:
      bbox_data = pickle.load(handle)
    for fid in fids_rel:
      fid_rel_seg = ped['map20fpsToRel'][ped['fids20'][fid]]
      obj_cls += bbox_data['obj_cls'][fid_rel_seg],
      obj_bbox += bbox_data['obj_bbox'][fid_rel_seg],
    # obj_cls = [bbox_data['obj_cls'][fid] for fid in fids_rel]
    # obj_bbox = [bbox_data['obj_bbox'][fid] for fid in fids_rel]

    img_paths = [self.img_path_format.format(ped['vid'], ped['seg'], ped['fids20'][fid]) for fid in fids_rel]

    ret = {
      'GT_act': GT_act,
      'GT_bbox': GT_bbox,
      'GT_pose': GT_pose,
      'obj_cls': obj_cls,
      'obj_bbox': obj_bbox,
      # 'obj_cls': obj_cls,
      # 'obj_bbox': obj_bbox,
      'fids': torch.tensor(np.array(fids_rel)),
      'img_paths': img_paths,
    }
    ped_crops, all_masks = [], []

    # only the first seq_len fids are input data
    fids_rel = fids_rel[:self.seq_len]

    if self.use_driver:
      frames = []
      for fid in fids_rel:
        img_path = self.img_path_format.format(ped['vid'], ped['seg'], fid) # +1 since fid is 0-based while the frame path starts at 1.
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img.transpose((2,0,1)) # i.e. img size = (3, 224, 224)
        frames += img,
      frames = torch.tensor(frames)
      ret['frames'] = frames

    if self.load_cache == 'masks':
      for i,fid in enumerate(fids_rel):
        # NOTE: fid for masks is 1-based.
        fcache = self.cache_format.format(self.split, pid, ped['fids20'][fid])

        # if not os.path.exists(fcache):
        #   # some frames are missing in side views
        #   self.invalid_pid += pid,
        #   return None

        if os.path.exists(fcache):
          # Load cache if file exists.
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
                print('STIP: n_objs mismatch (load cache)')
                pdb.set_trace()
        else:
          # Generate cache if file doesn't exist.
          ped_crop, cls_masks = self.get_ped_fid(ped, fid, pid)
          ped_crops += ped_crop,
          all_masks += cls_masks,
          if type(cls_masks) == dict:
            n_objs = len([each for val in cls_masks.values() for each in val])
          else:
            n_objs = len(cls_masks)
          if n_objs != len(obj_bbox[i]):
            print('STIP: n_objs mismatch (get_ped_fid)')
            pdb.set_trace()
      ped_crops = np.stack(ped_crops, 0)
      ped_crops = torch.tensor(ped_crops)

      ret['ped_crops'] = ped_crops
      ret['all_masks'] = all_masks
      return ret

    elif self.load_cache == 'feats':
      ped_feats = []
      ctxt_feats = []
      for fid in fids_rel:
        # NOTE: fid for feats is 0-based.
        fcache = self.cache_format.format(self.split, pid, ped['fids20'][fid])
        # pdb.set_trace()
        if not os.path.exists(fcache):
          # some frames are missing in side views
          pdb.set_trace()
          self.invalid_pid += pid,
          return None
        with open(fcache, 'rb') as handle:
          data = pickle.load(handle)
          ped_feat = data['ped_feats'] # shape: 1, 512
          ctxt_feat = data['ctxt_feats'] # shape: n_objs, 512

          ped_feats += ped_feat,
          ctxt_feats += ctxt_feat,
          # pdb.set_trace()
      ped_feats = torch.stack(ped_feats, 0)
      
      ret['ped_crops'] = ped_feats
      ret['all_masks'] = ctxt_feats
      return ret

    elif self.load_cache == 'pos':
      ret['ped_crops'] = torch.zeros([1,1,512])
      ret['all_masks'] = torch.zeros([1,1,512])
      return ret

    for fid in fids_rel:
      ped_crop, cls_masks = self.get_ped_fid(ped, fid, pid)
      ped_crops += ped_crop,
      all_masks += cls_masks,

    # shape: [n_frames, self.ped_crop_size[0], self.ped_crop_size[1]]

    ped_crops = np.stack(ped_crops)
    ped_crops = torch.Tensor(ped_crops)

    # print('time per item:', time.time()-t_start)
    ret['ped_crops'] = ped_crops
    ret['all_masks'] = all_masks
    return ret


  def get_ped_fid(self, ped, fid, pid):
      """
      Prepare ped_crop and obj masks for given ped and fid.
      """

      ped_pos = ped['pos_GT'][fid]
      if len(ped_pos) == 0:
        # if no bbox, take the entire frame
        x,y,w,h = 0,0,-1,-1
      else:
        x, y, w, h = ped_pos
        x, y, w, h = max(0, int(x)), max(0, int(y)), int(w), int(h)

      # pedestrian crop
      img_path = self.img_path_format.format(ped['vid'], ped['seg'], ped['fids20'][fid])
      img = cv2.imread(img_path)
      # img = cv2.resize(img, (224, 224))
      # img = img.transpose((2,0,1))
      try:
        ped_crop = img[y:y+h, x:x+w]
      except Exception as e:
        print(e)
        print('img_path:', img_path)
        print('x:{}, y:{}, w:{}, h:{}'.format(x, y, w, h))
        peb_crop = np.zeros(img.shape)
      ped_crop = cv2.resize(ped_crop, self.ped_crop_size)
      ped_crop = ped_crop.transpose((2,0,1))
      
      # obj masks
      fsegm = self.fsegm_format.format(ped['vid'], ped['seg'], ped['fids20'][fid])
      if not os.path.exists(fsegm):
        # no segm file for the frame
        if self.combine_method:
          # e.g. 'sum' or 'max'
          cls_masks = torch.zeros(self.mask_size)
        else:
          cls_masks = torch.zeros([0, self.mask_size[0], self.mask_size[1]])
      else:
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

      # NOTE: used to be 'self.save_cache_format'
      if self.cache_format:
        fcache = self.cache_format.format(self.split, pid, ped['fids20'][fid])
        os.makedirs(os.path.dirname(fcache), exist_ok=True)
        with open(fcache, 'wb') as handle:
          cache = {
            'ped_crops': ped_crop,
            'masks': cls_masks.data if self.collapse_cls else {cls:cls_masks[cls].data for cls in cls_masks},
          }
          pickle.dump(cache, handle)

      return ped_crop, cls_masks


