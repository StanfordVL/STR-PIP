import os
from glob import glob
import json
import cv2
import pdb

MACHINE = 'TRI'
MACHINE = 'CS'

fps_annot = 2
fps_png = 20
fps_tgt = 2 # target FPS
overRate = 1 # oversample rate for annots
downRate = 10 # downsample rate for png

if MACHINE == 'TRI':
  # on TRI instance
  stip_root = '/mnt/paralle/stip'
  annot_root = os.path.join(stip_root, 'annotation')
  # subdir_format_png: two placeholders: area (e.g. "ANN_conor1"), subdir (e.g. "00:15-00:41").
  subdir_format_png = os.path.join(stip_root, 'stip_instances', '{}/rgb/{}')
else:
  # on Stanford CS servers
  stip_root = '/vision2/u/bingbin/STR-PIP/STIP'
  annot_root = os.path.join(stip_root, 'annotations')
  subdir_format_png = os.path.join(stip_root, 'rgb', '{}/{}')

# drawing setting
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 4
TEXT_COLOR = (255, 255, 255)
TEXT_THICK = 2

def png_lookup(fpngs):
  png_map = {}
  for fpng in fpngs:
    fid = int(os.path.basename(fpng).split('.')[0])
    png_map[fid] = fpng
  return png_map


def visual_clip(fannot, sdarea, tgt_dir):
  # pngs = sorted([fpng for sdir in glob(png_formt.format(darea, '*')) for fpng in os.path.join(sdir, '*.png')])
  fpngs_all = sorted([fpng for fpng in glob(os.path.join(sdarea, '*.png'))])
  png_map = png_lookup(fpngs_all)

  os.makedirs(tgt_dir, exist_ok=True)

  annot = json.load(open(fannot, 'r'))
  frames_annot = annot['frames']
  for fid_annot in frames_annot:
    fid_tgt = [overRate*(int(fid_annot)-1)+i for i in range(overRate)]
    fid_png = [tfid*downRate for tfid in fid_tgt]
    fpngs = [png_map[pfid] for pfid in fid_png if pfid in png_map]
    if len(fpngs):
      print('# fpngs: {} / len(png_map): {}'.format(len(fpngs), len(png_map)))
      print('fid_annot:', fid_annot)
      print('fid_tgt:', fid_tgt)
      print('fid_png:', fid_png)
      print('fpngs:', fpngs)
      print()

    # draw bbox on png frames
    annot = frames_annot[fid_annot]
    for fpng in fpngs:
      img = cv2.imread(fpng)
      print('img size:', img.shape)
      if len(annot) == 0:
        cv2.putText(img, 'No obj in the current frame.', (100,100),
                    FONT, FONT_SCALE, TEXT_COLOR, TEXT_THICK)
      else:
        for obj in annot:
          cv2.rectangle(img, (int(obj['x1']), int(obj['y1'])), (int(obj['x2']), int(obj['y2'])), (0,255,0), 3)
          cv2.putText(img, '-'.join(obj['tags']), (int(obj['x1']), int(obj['y1'])),
                      FONT, FONT_SCALE, TEXT_COLOR, TEXT_THICK)
      tgt_fpng = fpng.replace(sdarea, tgt_dir)
      if tgt_fpng == fpng:
        print('Error saving drawn png: tgt_fpng == fpng')
        print('fpng:', fpng)
        pdb.set_trace()
      else:
        cv2.imwrite(tgt_fpng, img)

def visual_clip_wrapper():
  fannot = os.path.join(annot_root, '20170907_prolix_trial_ANN_hanh2-09-07-2017_15-44-07.concat.12fps.mp4.json')
  sdarea = subdir_format_png.format('ANN_hanh2', '00:16--00:27')
  if MACHINE == 'TRI':
    tgt_dir = sdarea.replace('/stip/', '/stip/tmp_vis/')
  else:
    tgt_dir = sdarea.replace('/rgb/' ,'/tmp_vis/')
  visual_clip(fannot, sdarea, tgt_dir)


if __name__ == "__main__":
  visual_clip_wrapper()
