from PIL import Image
import cv2
import numpy as np
import os
from glob import glob
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pickle
import random

from colormap import colormap

import pdb

FLIP = 1
TRANS = 0

ckpt_dir = '/sailhome/bingbin/crossing/ckpts/JAAD/graph_gru_seq30_pred30_lr1.0e-04_wd1.0e-05_bt16_posNone_branchboth_collapse0_combinepair_adjTypespatial_nLayers2_v4Feats_pedGRU_newCtxtGRU_3evalEpoch/'

cache_dir = os.path.join(ckpt_dir, 'test_graph_weights_epochbest_pred')
out_dir = os.path.join(ckpt_dir, 'vis_out')
os.makedirs(out_dir, exist_ok=True)

def tmp_vis_one_image():
  fpkls = sorted(glob(os.path.join(cache_dir, '*pkl')))
  for pi,fpkl in enumerate(fpkls):
    if pi < 70:
      continue
    if pi and pi%10 == 0:
      print("{} / {}".format(pi, len(fpkls)))

    with open(fpkl, 'rb') as handle:
      vid = pickle.load(handle)
    v_ws = vid['ws']
    v_bbox = vid['obj_bbox']
    img_names = vid['img_paths']

    for i in range(len(v_ws)):
      img_name = img_names[i]
      fid = os.path.basename(img_name).split('.')[0]
      out_name = os.path.join(out_dir, os.path.basename(fpkl).replace('.pkl', '_i{}_f{}.png'.format(i, fid)))
      ws = v_ws[i][-1] # take weights from the last graph layer
      vis_one_image(img_names[i], out_name, v_bbox[i], weights=ws)

def vis_one_image(im_name, fout, bboxes, dpi=200, weights=None):
  # masks: (N, 28, 28) ... masks for one frame
  if not len(bboxes):
    return

  im = cv2.imread(im_name)
  H, W, _ = im.shape
  color_list = colormap(rgb=True) / 255

  fig = plt.figure(frameon=False)
  fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.axis('off')
  fig.add_axes(ax)
  ax.imshow(im)

  mask_color_id = 0
  if weights is None:
    n_objs = masks.shape[0]
    obj_ids = range(n_objs)
  else:
    obj_ids = np.argsort(weights)

  ws = [0]
  for oid in obj_ids:
    x,y,w,h = bboxes[oid]
    mask = np.zeros([H, W])
    mask[x:x+w, y:y+h] = 1
    mask = mask.astype('uint8')
    if mask.sum() == 0:
      continue

    if weights is not None:
      ws += weights[oid],
    color_mask = color_list[mask_color_id % len(color_list), 0:3]
    mask_color_id += 1

    w_ratio = .4
    for c in range(3):
      color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio

    e_down = mask

    e_pil = Image.fromarray(e_down)
    e_pil_up = e_pil.resize((H, W) if TRANS else (W, H),Image.ANTIALIAS)
    e = np.array(e_pil_up)

    _, contour, hier = cv2.findContours(e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if len(contour) > 1:
      print('# contour:', len(contour))
    for c in contour:
      if FLIP:
        assert(c.shape[1] == 1 and c.shape[2] == 2), print('c.shape:', c.shape)
        for pid in range(c.shape[0]):
          c[pid][0][0], c[pid][0][1] = c[pid][0][1], c[pid][0][0]
      linewidth = 1.2
      alpha = 0.5
      if oid == obj_ids[-1]:
        # most probable obj
        edgecolor=(1,0,0,1) # 'r'
      else:
        edgecolor=(1,1,1,1) # 'w'
      if weights is not None:
        linewidth *= (4 ** weights[oid])
        alpha /= (4 ** weights[oid])

      polygon = Polygon(
        c.reshape((-1, 2)),
        fill=True, facecolor=(color_mask[0], color_mask[1], color_mask[2], alpha),
        edgecolor=edgecolor, linewidth=linewidth,
        )
      xy = polygon.get_xy()

      ax.add_patch(polygon)

  fig.savefig(fout.replace('.jpg', '_{:.3f}.jpg'.format(max(ws))), dpi=dpi)
  plt.close('all')


if __name__ == '__main__':
  # tmp_wrapper()
  tmp_vis_one_image()

