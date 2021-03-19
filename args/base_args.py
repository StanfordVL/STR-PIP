import argparse
import os
from datetime import datetime


class BaseArgs:
  def __init__(self):
    self.is_train = None
    self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # hardware
    self.parser.add_argument('--n-workers', type=int, default=8, help='number of threads')
    self.parser.add_argument('--device', type=int, default=0, help='visible GPU ids, separated by comma')

    # model
    self.parser.add_argument('--model', type=str)
    self.parser.add_argument('--reg-smooth', type=str, default='none', choices=['none', 'l1', 'l2', 'hinge'])
    self.parser.add_argument('--reg-lambda', type=float, default=0.1,
                             help='loss += lambda * reg_loss')
    self.parser.add_argument('--conv-dim', type=int, default=512)
    self.parser.add_argument('--branch', type=str, default='both', choices=['both', 'ped', 'ctxt', 'graph'],
                             help='What feat to use for Concat model.')
    self.parser.add_argument('--n-acts', type=int, default=1)
    self.parser.add_argument('--n-driver-acts', type=int, default=5)
    self.parser.add_argument('--seq-len', type=int, default=30)
    self.parser.add_argument('--predict', type=int, default=0,
                             help='Whether to predict future crossing prob.')
    self.parser.add_argument('--pred-seq-len', type=int, default=10)
    self.parser.add_argument('--predict-k', type=int, default=0,
                             help="Predict label for the kth frame in the future.")
    self.parser.add_argument('--ped-crop-size', type=tuple, default=(224, 224))
    self.parser.add_argument('--mask-size', type=tuple, default=(224, 224))    
    self.parser.add_argument('--collapse-cls', type=int, default=0,
                             help='Whether to merge the classes. If 1 then each item in masks is a dict keyed by cls, otherwise a list.')
    self.parser.add_argument('--combine-method', type=str, default='',
                             help="Whether to combine obj_masks into one. Options include 'sum' and 'max'.")
    self.parser.add_argument('--use-bbox', type=int, default=0,
                             help="Whether to use GT bbox position as part of the features.")
    self.parser.add_argument('--use-signal', type=int, default=0,
                             help="Whether to use GT pedestrian signal (handwave & looking) as part of the features.")
    self.parser.add_argument('--use-driver', type=int, default=0,
                             help="Whether to use the prediction on driver's behavior.")
    self.parser.add_argument('--use-gru', type=int, default=0,
                             help='ConcatModel: whether to use GRU on frame_feats')
    self.parser.add_argument('--use-trn', type=int, default=0,
                             help="Use TRN to aggregate temporal info & predict for the next frame.")
    self.parser.add_argument('--ped-gru', type=int, default=0,
                             help="Whether to use a GRU on pedestrian features.")
    self.parser.add_argument('--ctxt-gru', type=int, default=0,
                             help="Whether to use a GRU on collapsed (avg) ctxt features.")
    self.parser.add_argument('--ctxt-node', type=int, default=0,
                             help="Whether to include a context node w/ GRU connection.")
    self.parser.add_argument('--frame-gru', type=int, default=0,
                             help="Whether to connect center node (frame) in loc graph.")
    self.parser.add_argument('--node-gru', type=int, default=0,
                             help="Whether to add temporal connection on aggregated nodes in loc graph.")
    self.parser.add_argument('--pos-mode', type=str, default='none', choices=['none', 'bbox', 'center', 'both', 'height'],
                             help="What position info of the pedestrain that gets fed into the model.")
    self.parser.add_argument('--use-act', type=int, default=0, help="Whether to use act labels (GT or predicted) for prediction.")
    self.parser.add_argument('--use-gt-act', type=int, default=0,
                             help="Whether to include GT act labels in model_pos.")
    self.parser.add_argument('--use-pose', type=int, default=0,
                             help="Whether to use pose (i.e. x/y coor of 9 joints) in the model.")
    self.parser.add_argument('--use-obj-cls', type=int, default=0,
                             help="Whether to use obj_cls in 'spatial' relation for graph.")
    self.parser.add_argument('--rand-test', type=int, default=0,
                             help='Whether to randomly sample during testing.')
    self.parser.add_argument('--n-layers', type=int, default=2,
                             help="Number of graph conv layers.")
    self.parser.add_argument('--diff-layer-weight', type=int, default=0,
                             help="Whether to use diff embedding layers in graph conv.")
    self.parser.add_argument('--adj-type', type=str, default='uniform',
                             choices=['uniform', 'prior', 'inner', 'cosine', 'spatial', 'all', 'spatialOnly', 'random'])

    # data
    self.parser.add_argument('--view', type=str, default='center', choices=['center', 'all', 'left', 'right'],
                             help='Choose to use only the center view ("center"), or also the side views ("all")')
    self.parser.add_argument('--split', type=str, required=True)
    self.parser.add_argument('--dset-name', type=str, default='JAAD')
    self.parser.add_argument('--annot-ped-format', type=str,
                             # default='/sailhome/bingbin/STR-PIP/datasets/annot_{}_ped.pkl',
                             default='/sailhome/ajarno/STR-PIP/datasets/annot_{}_ped_withTag_sanityWithPose.pkl',
                             help='Format for the pkl file for parsed pedestrian annotations.')
    self.parser.add_argument('--annot-loc-format', type=str,
                             default='/sailhome/ajarno/STR-PIP/datasets/annot_{}_loc.pkl',
                             help='Format for the pkl file for parsed location-centric annotations.')
    self.parser.add_argument('--img-path-format', type=str,
                             default='/sailhome/ajarno/STR-PIP/datasets/JAAD_dataset/JAAD_clip_images/video_{:04d}.mp4/{:d}.jpg')
    self.parser.add_argument('--fsegm-format', type=str,
                             default='/sailhome/ajarno/STR-PIP/datasets/JAAD_instance_segm/video_{:04d}/{:08d}_segm.npy')
    # self.parser.add_argument('--fsegm-format-left', type=str, default='')
    # self.parser.add_argument('--fsegm-format-right', type=str, default='')
    self.parser.add_argument('--driver-act-format', type=str,
                             default='/sailhome/ajarno/STR-PIP/datasets/JAAD_dataset/JAAD_behavioral_encode/video_{:04d}/Driver.npy')
    self.parser.add_argument('--save-cache-format', type=str, default='')
    # self.parser.add_argument('--save-cache-format-left', type=str, default='')
    # self.parser.add_argument('--save-cache-format-right', type=str, default='')
    self.parser.add_argument('--load-cache', type=str, default='none',
                             help='Whether/what to load cached dataset.')
    self.parser.add_argument('--cache-format', type=str, default='')
    # self.parser.add_argument('--cache-format-left', type=str, default='')
    # self.parser.add_argument('--cache-format-right', type=str, default='')
    self.parser.add_argument('--cache-obj-bbox-format', type=str,
                             default='/sailhome/ajarno/STR-PIP/datasets/cache/obj_bbox_merged/vid{:08d}.pkl',
                             help='Format for pkl files w/ keys: "obj_cls" and "obj_bbox".')
    # self.parser.add_argument('--cache-obj-bbox-format-left', type=str, default='')
    # self.parser.add_argument('--cache-obj-bbox-format-right', type=str, default='')

    # ckpt and logging
    self.parser.add_argument('--ckpt-dir', type=str, default='/sailhome/ajarno/STR-PIP/ckpts',
                             help='the directory that contains all checkpoints')
    self.parser.add_argument('--ckpt-name', type=str, default='ckpt', help='checkpoint name')
    self.parser.add_argument('--ckpt-path', type=str, default='ckpt_path', help='placeholder for the real name.')
    self.parser.add_argument('--pretrained-path', type=str, default='',
                             help='Pretrained model to load for training.')
    self.parser.add_argument('--log-every', type=int, default=50, help='log every x steps')
    self.parser.add_argument('--save-every', type=int, default=10, help='save every x epochs')
    self.parser.add_argument('--evaluate-every', type=int, default=-1, help='evaluate on val set every x epochs')
    self.parser.add_argument('--extract-feats-dir', type=str, default='',
                             help="Dir at which extracted features from backbone (i.e. 'ped_encoder' and 'ctxt_encoder') will be saved. If empty str then don't save.")
    self.parser.add_argument('--save-output', type=int, default=0, help='Save output/GT labels every x steps. If x then do not save.')
    self.parser.add_argument('--save-output-format', type=str, default='', help='Path format if saving output/GT labels.')


  def parse(self):
    opt, _ = self.parser.parse_known_args()
    print("I AM HERE")
    # opt.is_train, opt.split = self.is_train, self.split
    opt.is_train = self.is_train
    if opt.is_train:
      if opt.model == 'ConcatModel':
        model_name = 'concat'
      else:
        model_name = opt.model
      if opt.use_gru:
        model_name += '_gru'
      ckpt_name = '{:s}_seq{}{}_lr{:.01e}_wd{:.01e}_bt{:d}_pos{}_{:s}'.format(
                  model_name, opt.seq_len,
                  '_pred{}'.format(opt.pred_seq_len) if opt.predict else '',
                  opt.lr_init, opt.wd, opt.batch_size,
                  opt.pos_mode[0].upper()+opt.pos_mode[1:],
                  opt.ckpt_name)
    else:
      ckpt_name = opt.ckpt_name
    opt.ckpt_path = os.path.join(opt.ckpt_dir, opt.dset_name, ckpt_name)

    if opt.is_train and os.path.exists(opt.ckpt_path):
      print('Warning: ckpt dir exists:', opt.ckpt_path)
      overwrite = input('Overwrite (y/N)?')
      if 'y' not in overwrite and 'Y' not in overwrite:
        proceed = input('Proceed with timestamp (Y/n)?')
        if 'n' not in overwrite and 'N' not in overwrite:
          now = datetime.now()
          timestamp = '_{:02d}m{:02d}d{:02d}h{:02d}m{:02d}s'.format(
            now.month, now.day, now.hour, now.minute, now.second)
          opt.ckpt_path += timestamp
        else:
          print("Do not overwrite or proceed --> Exiting. Bye~!")
          exit(0)

    log = ['Arguments: ']
    for k, v in sorted(vars(opt).items()):
      log.append('{}: {}'.format(k, v))

    return opt, log

