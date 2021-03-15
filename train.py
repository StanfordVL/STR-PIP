import os
import numpy as np
import pickle
import copy
import sys, traceback, code
import torch

import data
import models
import utils
from test import evaluate

import pdb

import wandb

N_EVAL_EPOCHS = 3

opt, logger = utils.build(is_train=True)
with open(os.path.join(opt.ckpt_path, 'opt.pkl'), 'wb') as handle:
  pickle.dump(opt, handle)

tags = [opt.model, opt.branch, 'w/ bbox' if opt.use_bbox else 'no bbox', 'seq{}'.format(opt.seq_len)]
if opt.model == 'graph':
  tags += ['{} layer'.format(opt.n_layers), opt.adj_type]
  tags += ['diff {}'.format(opt.diff_layer_weight)]
  tags += [opt.adj_type]
if opt.model == 'pos' or opt.pos_mode != 'none':
  tags += opt.pos_mode,
if opt.predict:
  tags += 'pred{}'.format(opt.pred_seq_len),
if opt.predict_k:
  tags += 'pred_k{}'.format(opt.predict_k),
if opt.ped_gru:
  tags += 'pedGRU',
if opt.ctxt_gru:
  tags += 'ctxtGRU',
if opt.ctxt_node:
  tags += 'ctxtNode',
if opt.load_cache == 'none':
  tags += 'cacheNone',
if opt.load_cache == 'masks':
  tags += 'cacheMasks',
if opt.load_cache == 'feats':
  tags += 'cacheFeats',
if opt.use_driver:
  tags += 'driver',
tags += '{}evalEpochs'.format(N_EVAL_EPOCHS),
wandb.init(
  project='crossing',
  tags=tags)

opt_dict = vars(opt)
print('Options:')
for key in sorted(opt_dict):
  print('{}: {}'.format(key, opt_dict[key]))


# train
print(opt)
train_loader = data.get_data_loader(opt)
print('Train dataset: {}'.format(len(train_loader.dataset)))

# val
# val_opt = copy.deepcopy(opt)
val_opt, _ = utils.build(is_train=False)
val_opt.split = 'test'
val_opt.slide = 0
val_opt.is_train = False
val_opt.rand_test = True
val_opt.batch_size = 1
val_opt.slide = 0
val_loader = data.get_data_loader(val_opt)
print('Val dataset: {}'.format(len(val_loader.dataset)))

model = models.get_model(opt)
# model = model.to('cuda:{}'.format(opt.device))
if opt.pretrained_path and os.path.exist(opt.pretrained_path):
  print('Loading model from', opt.pretrained_path)
  model.load(opt.pretrained_path)
model = model.to('cuda:0')
wandb.watch(model)

if opt.load_ckpt_dir != '':
  ckpt_dir = os.path.join(opt.ckpt_dir, opt.dset_name, opt.load_ckpt_dir)
  assert os.path.exists(ckpt_dir)
  logger.print('Loading checkpoint from {}'.format(ckpt_dir))
  model.load(ckpt_dir, opt.load_ckpt_epoch)

opt.n_epochs = max(opt.n_epochs, opt.n_iters // len(train_loader))
logger.print('Total epochs: {}'.format(opt.n_epochs))


def train():
  best_eval_acc = 0
  best_eval_loss = 10
  best_epoch = -1

  if val_opt.predict or val_opt.predict_k:
    # pred
    best_eval_acc_pred = 0
    best_eval_loss_pred = 10
    best_epoch_pred = -1
    # last
    best_eval_acc_last = 0
    best_eval_loss_last = 10
    best_epoch_last = -1

  for epoch in range(opt.start_epoch, opt.n_epochs):
    model.setup()
    print('Train epoch', epoch)
    model.update_hyperparameters(epoch)
  
    losses = []
    for step, data in enumerate(train_loader):
      # break
      if epoch == 0:
        torch.cuda.empty_cache()
        # break
      loss = model.step_train(data)
      losses += loss,

      torch.cuda.empty_cache()
  
      if step % opt.log_every == 0:
        print('avg loss:', sum(losses) / len(losses))
        wandb.log({"Train loss:":sum(losses) / len(losses)})
        losses = []
  
    # Evaluate on val set
    if opt.evaluate_every > 0 and (epoch + 1) % opt.evaluate_every == 0:
      result_det, result_pred, result_last = evaluate(model, val_loader, val_opt, n_eval_epochs=N_EVAL_EPOCHS)
      eval_acc_frame, eval_acc_clip, eval_acc_cross, eval_acc_non_cross, eval_loss = result_det
      if eval_acc_frame > best_eval_acc:
        best_eval_acc = eval_acc_frame
        best_eval_loss = eval_loss
        best_epoch = epoch+1
        model.save(opt.ckpt_path, best_epoch, 'best_det')
      wandb.log({
        'eval_acc_frame':eval_acc_frame, 'eval_acc_clip':eval_acc_clip,
        'eval_acc_cross':eval_acc_cross, 'eval_acc_non_cross':eval_acc_non_cross,
        'eval_loss':eval_loss,
        'best_eval_acc': best_eval_acc, 'best_eval_loss':best_eval_loss, 'best_epoch':best_epoch})

      if val_opt.predict or val_opt.predict_k:
        # pred
        eval_acc_frame, eval_acc_clip, eval_acc_cross, eval_acc_non_cross, eval_loss = result_pred
        if eval_acc_frame > best_eval_acc_pred:
          best_eval_acc_pred = eval_acc_frame
          best_eval_loss_pred = eval_loss
          best_epoch_pred = epoch+1
          model.save(opt.ckpt_path, best_epoch_pred, 'best_pred')
        wandb.log({
          'eval_acc_frame_pred':eval_acc_frame, 'eval_acc_clip_pred':eval_acc_clip,
          'eval_acc_cross_pred':eval_acc_cross, 'eval_acc_non_cross_pred':eval_acc_non_cross,
          'eval_loss_pred':eval_loss,
          'best_eval_acc_pred': best_eval_acc_pred, 'best_eval_loss_pred':best_eval_loss_pred, 'best_epoch_pred':best_epoch_pred})
        # last
        eval_acc_frame, eval_acc_clip, eval_acc_cross, eval_acc_non_cross, eval_loss = result_last
        if eval_acc_frame > best_eval_acc_last:
          best_eval_acc_last = eval_acc_frame
          best_eval_loss_last = eval_loss
          best_epoch_last = epoch+1
          model.save(opt.ckpt_path, best_epoch_last, 'best_last')
        wandb.log({
          'eval_acc_frame_last':eval_acc_frame, 'eval_acc_clip_last':eval_acc_clip,
          'eval_acc_cross_last':eval_acc_cross, 'eval_acc_non_cross_last':eval_acc_non_cross,
          'eval_loss_last':eval_loss,
          'best_eval_acc_last': best_eval_acc_last, 'best_eval_loss_last':best_eval_loss_last, 'best_epoch_last':best_epoch_last})

  
    # Save model checkpoints
    if (epoch + 1) % opt.save_every == 0 and epoch >= 0 or epoch == opt.n_epochs - 1:
      model.save(opt.ckpt_path, epoch+1)


try:
  train()
except Exception as e:
  print(e)
  typ, vacl, tb = sys.exc_info()
  traceback.print_exc()
  last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
  frame = last_frame().tb_frame
  ns = dict(frame.f_globals)
  ns.update(frame.f_locals)
  code.interact(local=ns)

