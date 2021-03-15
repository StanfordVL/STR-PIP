import os
import numpy as np
import pickle
import sys, traceback, code
import torch

import data
import models
import utils

import pdb


def evaluate(model, dloader, opt, n_eval_epochs=3):
  print('Begin to evaluate')
  model.eval()

  if opt.collect_A:
    os.makedirs(os.path.dirname(opt.save_As_format), exist_ok=True)
  
  acc_det = {
    'frames': 0,
    'correct_frames': 0,
    'clips': 0,
    'correct_clips': 0,
    'cross': 0,
    'non_cross': 0,
    'correct_cross': 0,
    'correct_non_cross': 0,
    'probs': None,
    'loss': 0,
  }
  label_out = [] # labels output by the model
  label_GT = [] # GT labels (crossing)
  label_prob = []

  if opt.predict or opt.predict_k:
    acc_pred = {key:0 for key in acc_det}
    acc_last = {key:0 for key in acc_det}

  def helper_update_metrics(ret, acc):
    n_frames, n_correct_frames, n_clips, n_correct_clips = ret[:4]
    n_cross, n_non_cross, n_correct_cross, n_correct_non_cross = ret[4:8]
    probs = ret[8]
    loss = ret[9]
    preds = ret[10] # (B, T)
    crossing = ret[11] # (B, T)

    acc['frames'] += n_frames
    acc['correct_frames'] += n_correct_frames
    acc['clips'] += n_clips
    acc['correct_clips'] += n_correct_clips
    acc['cross'] += n_cross
    acc['non_cross'] += n_non_cross
    acc['correct_cross'] += n_correct_cross
    acc['correct_non_cross'] += n_correct_non_cross
    if acc['probs'] is None:
      acc['probs'] = probs
    else:
      acc['probs'] += probs
    acc['loss'] += loss

    return acc

  def helper_report_metrics(acc):
    if acc['probs'] is None:
      return 0, 0, 0, 0, 100

    acc_frame = acc['correct_frames'] / max(1, acc['frames'])
    acc_clip = acc['correct_clips'] / max(1, acc['clips'])
    acc_cross = acc['correct_cross'] / max(1, acc['cross'])
    acc_non_cross = acc['correct_non_cross'] / max(1, acc['non_cross'])
    avg_probs = acc['probs'] / max(1, acc['clips'])
    avg_loss = acc['loss'] / max(1, acc['frames'])
    print('Accuracy: frame:{:.5f}\t/ clip:{:.5f}'.format(acc_frame, acc_clip))
    print('Recall: cross:{:.5f}\t/ non-cross:{:.5f}'.format(acc_cross, acc_non_cross))
    print('Probs:', ' / '.join(['{}:{:.1f}'.format(i, each.item()*100) for i,each in enumerate(avg_probs)]))
    print('Loss: {:.3f}'.format(avg_loss))
    return acc_frame, acc_clip, acc_cross, acc_non_cross, avg_loss

  with torch.no_grad():
    for eid in range(n_eval_epochs):
      for step, data in enumerate(dloader):
        ret_det, ret_pred, ret_last, As = model.step_test(data, slide=opt.slide, collect_A=opt.collect_A)

        if ret_det is not None:
          acc_det = helper_update_metrics(ret_det, acc_det)

        if opt.predict or opt.predict_k:
          acc_pred = helper_update_metrics(ret_pred, acc_pred)
          acc_last = helper_update_metrics(ret_last, acc_last)
    
          if opt.save_output > 0 and ret_det is not None:
            curr_out = torch.cat([ret_det[10], ret_pred[10], ret_last[10]], -1)
            curr_GT = torch.cat([ret_det[11], ret_pred[11], ret_last[11]], -1)
            curr_prob = torch.cat([ret_det[8], ret_pred[8], ret_last[8]])
            label_out += curr_out,
            label_GT += curr_GT,
            label_prob += curr_prob,
        elif opt.save_output > 0 and ret_det is not None:
          label_out += ret_det[10],
          label_GT += ret_det[11],
          label_prob += ret_det[8],

        if As is not None:
          data = {
            'As': As,
            'fids': data['fids'],
            'img_paths': data['img_paths'],
            'probs': ret_pred[8], # 1D tensor of size T (avg over B)
          }
          with open(opt.save_As_format.format(step, eid), 'wb') as handle:
            pickle.dump(data, handle)

        if opt.save_output and (step+1)%opt.save_output == 0 and False:
          label_out = torch.cat(label_out, 0).numpy()
          label_GT = torch.cat(label_GT, 0).numpy()
          label_prob = torch.cat(label_prob, 0).numpy()
          with open(opt.save_output_format.format(step), 'wb') as handle:
            pickle.dump({'out':label_out, 'GT': label_GT, 'prob': label_prob}, handle)
          label_out = []
          label_GT = []
          label_prob = []

        torch.cuda.empty_cache()
    
  if opt.save_output_format:
    label_out = torch.cat(label_out, 0).numpy()
    label_GT = torch.cat(label_GT, 0).numpy()
    label_prob = torch.cat(label_prob, 0).numpy()
    with open(opt.save_output_format.format('all'), 'wb') as handle:
      pickle.dump({'out':label_out, 'GT': label_GT, 'prob': label_prob}, handle)

  print('Detection:')
  result_det = helper_report_metrics(acc_det)
  if opt.predict or opt.predict_k:
    print('Prediction:')
    result_pred = helper_report_metrics(acc_pred)
    result_last = helper_report_metrics(acc_last)
    print()
    return result_det, result_pred, result_last

  print()
  return result_det, None, None


def extract_feats(model, dloader, extract_feats_dir, seq_len=30):
  print('Begin to extract')
  model.eval()
  
  n_peds = len(dloader)
  print('n_peds:', n_peds)

  for pid in range(0, n_peds):
    ped = dloader.dataset.peds[pid]
    if 'frame_end' in ped:
      # JAAD setting
      n_frames = ped['frame_end'] - ped['frame_start'] + 1
      fid_range = range(ped['frame_start'], ped['frame_end']+1)
      fid_display = list(fid_range)
    elif 'fids20' in ped:
      # STIP setting
      n_frames = len(ped['fids20'])
      fid_range = range(n_frames)
      fid_display = ped['fids20']
    else:
      print("extract_feats: missing/unexpected keys... o_o")
      pdb.set_trace()

    for fid,fid_dis in zip(fid_range, fid_display):
      print('pid:{} / fid:{}'.format(pid, fid))
      item = dloader.dataset.__getitem__(pid, fid_start=fid)
      ped_crops, masks, act = item['ped_crops'], item['all_masks'], item['GT_act']
      # print('masks[0][1]:', masks[0][1].shape)
      ped_feats, ctxt_feats, ctxt_cls = model.extract_feats(ped_crops, masks, pid)
      
      feat_path = os.path.join(extract_feats_dir, 'ped{}_fid{}.pkl'.format(pid, fid_dis))
      with open(feat_path, 'wb') as handle:
        feats = {
          'ped_feats': ped_feats.cpu(), # shape: 1, 512
          'ctxt_feats': ctxt_feats.cpu(), # shape: n_objs, 512
          'ctxt_cls': torch.tensor(ctxt_cls)
        }
        pickle.dump(feats, handle)

      del ped_feats
      del ctxt_feats

      torch.cuda.empty_cache()

    if pid % opt.log_every == 0:
      print('pid', pid)

def extract_feats_loc(model, dloader, extract_feats_dir, seq_len=1):
  print('Begin to extract')
  model.eval()
  
  n_vids = len(dloader)
  print('n_vids:', n_vids)

  for vid in range(0, n_vids):
    key = dloader.dataset.vids[vid]
    annot = dloader.dataset.annots[key]

    for fid in range(len(annot['act'])):
      print('vid:{} / fid:{}'.format(vid, fid))
      feat_path = os.path.join(extract_feats_dir, 'vid{}_fid{}.pkl'.format(vid, fid))
      if os.path.exists(feat_path):
        continue

      item = dloader.dataset.__getitem__(vid, fid_start=fid)
      ped_crops, masks, act = item['ped_crops'], item['all_masks'], item['GT_act']
      # print('masks[0][1]:', masks[0][1].shape)
      ped_feats, ctxt_feats, ctxt_cls = model.extract_feats(ped_crops, masks)
      
      with open(feat_path, 'wb') as handle:
        feats = {
          'ped_feats': ped_feats[0].cpu(), # shape: 1, 512
          'ctxt_feats': ctxt_feats.cpu(), # shape: n_objs, 512
          'ctxt_cls': torch.tensor(ctxt_cls)
        }
        pickle.dump(feats, handle)

      del ped_feats
      del ctxt_feats

      torch.cuda.empty_cache()

    if vid % opt.log_every == 0:
      print('vid', vid)


if __name__ == '__main__':
  opt, logger = utils.build(is_train=False)

  dloader = data.get_data_loader(opt)
  print('{} dataset: {}'.format(opt.split, len(dloader.dataset)))
  
  model = models.get_model(opt)
  print('Got model')
  if opt.which_epoch == -1:
    model_path = os.path.join(opt.ckpt_path, 'best_pred.pth')
  else:
    model_path = os.path.join(opt.ckpt_path, '{}.pth'.format(opt.which_epoch))
  if os.path.exists(model_path):
    # NOTE: if path not exists, then using backbone weights from ImageNet-pretrained model
    model.load(model_path)
    print('Model loaded:', model_path)
  else:
    print('Model does not exists:', model_path)
  model = model.to('cuda:0')
  
  try:
    if opt.mode == 'evaluate':
      evaluate(model, dloader, opt)
    elif opt.mode == 'extract':
      assert(opt.batch_size == 1)
      assert(opt.seq_len == 1)
      assert(opt.predict == 0)

      print('Saving at', opt.extract_feats_dir)
      os.makedirs(opt.extract_feats_dir, exist_ok=True)
      if 'loc' in opt.model:
        extract_feats_loc(model, dloader, opt.extract_feats_dir, opt.seq_len)
      else:
        extract_feats(model, dloader, opt.extract_feats_dir, opt.seq_len)
  except Exception as e:
    print(e)
    typ, vacl, tb = sys.exc_info()
    traceback.print_exc()
    last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
    frame = last_frame().tb_frame
    ns = dict(frame.f_globals)
    ns.update(frame.f_locals)
    code.interact(local=ns)

