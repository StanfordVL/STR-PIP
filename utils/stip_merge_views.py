import os
from glob import glob
import numpy as np
import pickle
import torch
import pdb


if False:
  # Objs
  print('Objects')
  root = '/vision/group/prolix/processed'
  center_root = os.path.join(root, 'center', 'obj_bbox_20fps')
  left_root = os.path.join(root, 'left', 'obj_bbox_20fps')
  right_root = os.path.join(root, 'right', 'obj_bbox_20fps')
  merged_root = os.path.join(root, 'all', 'obj_bbox_20fps')
  os.makedirs(merged_root, exist_ok=True)
  
  def merged_obj_loader(f):
    with open(f, 'rb') as handle:
      data = pickle.load(handle)
  
    obj_cls, obj_bbox = [], []
    for (cs, bs) in zip(data['obj_cls'], data['obj_bbox']):
      # bs: [xmin, ymin, w, h]
      curr_cls, curr_bbox = [], []
      for i in range(len(bs)):
        if bs[i, 3] != 0 and bs[i,2] != 0:
          # keep only non-empty bbox
          curr_cls += cs[i],
          curr_bbox += bs[i],
      obj_cls += np.array(curr_cls),
      obj_bbox += np.array(curr_bbox),
  
    return obj_cls, obj_bbox
  
  segs = sorted(glob(os.path.join(center_root, '*pkl')))
  for seg in segs:
    fl = seg.replace(center_root, left_root).replace('pkl', 'pkl.pkl')
    fr = seg.replace(center_root, right_root).replace('pkl', 'pkl.pkl')
    if not os.path.exists(fl) or not os.path.exists(fr):
      continue
    fout = seg.replace(center_root, merged_root) 
    # if os.path.exists(fout):
    #   continue
  
    with open(seg, 'rb') as handle:
      data = pickle.load(handle)
    with open(fl, 'rb') as handle:
      data_l = pickle.load(handle)
    with open(fr, 'rb') as handle:
      data_r = pickle.load(handle)
    assert(len(data) == len(data_l))
    assert(len(data) == len(data_r))
    data_all = {}
    for cls in data:
      data_l[cls] = [[x-1216, y, h, w] for (x,y,h,w) in data_l[cls]]
      data_r[cls] = [[x+1216, y, h, w] for (x,y,h,w) in data_l[cls]]
      data_all[cls] = data[cls] + data_l[cls] + data_r[cls]
    with open(fout, 'wb') as handle:
      pickle.dump(data_all, handle)

    # obj_cls_c, obj_bbox_c = obj_loader(seg)
    # obj_cls_l, obj_bbox_l = obj_loader(fl)
    # obj_cls_r, obj_bbox_r = obj_loader(fr)
  
    # obj_cls = [np.concatenate([c,l,r], 0) for (c,l,r) in zip(obj_cls_c, obj_cls_l, obj_cls_r)]
    # obj_bbox = []
    # for (c,l,r) in zip(obj_bbox_c, obj_bbox_l, obj_bbox_r):
    #   valid = [each for each in [c,l,r] if len(each.shape) > 1]
    #   if valid:
    #     a = np.concatenate(valid, 0)
    #   else:
    #     a = np.array([])
    #   obj_bbox += a,
    # # obj_bbox = [np.concatenate([c,l,r], 0) for (c,l,r) in zip(obj_bbox_c, obj_bbox_l, obj_bbox_r)]
    # 
    # with open(fout, 'wb') as handle:
    #   pickle.dump({'obj_cls':obj_cls, 'obj_bbox':obj_bbox}, handle)

# Masks
if False:
  print('Masks')
  root = '/vision/group/prolix/processed'
  center_root = os.path.join(root, 'cache/center')
  left_root = os.path.join(root, 'cache/left')
  right_root = os.path.join(root, 'cache/right')
  merged_root = os.path.join(root, 'cache/all')
  os.makedirs(merged_root, exist_ok=True)
  
  for split in ['train']:
    split_root = os.path.join(merged_root, split)
    os.makedirs(split_root, exist_ok=True)
  
    caches = sorted(glob(os.path.join(center_root, split, '*pkl')))
    for fc in caches:
      fl = fc.replace(center_root, left_root)
      fr = fc.replace(center_root, right_root)
      if not os.path.exists(fl) or not os.path.exists(fr):
        continue
      fout = fc.replace(center_root, merged_root) 
      # if os.path.exists(fout):
      #   continue
  
      with open(fc, 'rb') as handle:
        data = pickle.load(handle)
      with open(fl, 'rb') as handle:
        data_l = pickle.load(handle)
      with open(fr, 'rb') as handle:
        data_r = pickle.load(handle)

      data_out = {'ped_crops': data['ped_crops']}
      merged_masks = {k:[] for k in range(1,5)}
      keys = list(merged_masks.keys())
      for k in keys:
        if k in data['masks']:
          merged_masks[k] += data['masks'][k],
        if k in data_l['masks']:
          merged_masks[k] += data_l['masks'][k],
        if k in data_r['masks']:
          merged_masks[k] += data_r['masks'][k],
        if len(merged_masks[k]) == 0:
          merged_masks.pop(k)
        else:
          merged_masks[k] = torch.cat(merged_masks[k], 0)
      data_out['masks'] = merged_masks
  
      with open(fout, 'wb') as handle:
        pickle.dump(data_out, handle)

# Feats
if True:
  print('Feats')
  root = '/vision/group/prolix/processed/cache/'
  center_root = os.path.join(root, 'center/STIP_conv_feats')
  left_root = os.path.join(root, 'left/STIP_conv_feats')
  right_root = os.path.join(root, 'right/STIP_conv_feats')
  merged_root = os.path.join(root, 'all/STIP_conv_feats')
  os.makedirs(merged_root, exist_ok=True)
  
  for split in ['train']:
    caches = sorted(glob(os.path.join(center_root, "concat*", split, '*pkl')))
    split_root = os.path.dirname(caches[0]).replace(center_root, merged_root)
    os.makedirs(split_root, exist_ok=True)
  
    for fc in caches:
      fl = fc.replace(center_root, left_root)
      fr = fc.replace(center_root, right_root)
      if not os.path.exists(fl) or not os.path.exists(fr):
        print('Skipping')
        continue
      fout = fc.replace(center_root, merged_root) 
      # pdb.set_trace()
      #  if os.path.exists(fout):
      #   continue
  
      with open(fc, 'rb') as handle:
        data = pickle.load(handle)
      with open(fl, 'rb') as handle:
        data_l = pickle.load(handle)
      with open(fr, 'rb') as handle:
        data_r = pickle.load(handle)
      
      data_out = {'ped_feats': data['ped_feats']}
      merged_feats = []
      if data['ctxt_feats'].shape[0] != 1 or data['ctxt_feats'].sum() != 0:
        merged_feats += data['ctxt_feats'],
      if data_l['ctxt_feats'].shape[0] != 1 or data_l['ctxt_feats'].sum() != 0:
        merged_feats += data_l['ctxt_feats'],
      if data_r['ctxt_feats'].shape[0] != 1 or data_r['ctxt_feats'].sum() != 0:
        merged_feats += data_r['ctxt_feats'],
      merged_feats = torch.cat(merged_feats, 0)
      merged_cls = torch.cat([data['ctxt_cls'], data_l['ctxt_cls'], data_r['ctxt_cls']], 0)
      data_out['ctxt_feats'] = merged_feats
      data_out['ctxt_cls'] = merged_cls
  
      with open(fout, 'wb') as handle:
        pickle.dump(data_out, handle)
