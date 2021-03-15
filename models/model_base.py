import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from models.backbone.resnet_based import resnet_backbone



class BaseModel(nn.Module):
  def __init__(self, opt):
    super(BaseModel, self).__init__()
    # self.device = 'cuda:' + str(opt.device)
    self.device = 'cuda:0'
    self.dtype = torch.cuda.FloatTensor
    self.ped_encoder = resnet_backbone()
    self.ctxt_encoder = resnet_backbone()

    if opt.model == 'graph' and opt.load_cache == 'masks':
      # freeze the first 2 layers to reduce memory consumption
      for name, param in self.ped_encoder.named_parameters():
        if 'layer1' in name or 'layer2' in name or name=='conv1.weight' or name=='bn1.weight' or name=='bn1.bias':
          param.requires_grad = False
      for name, param in self.ctxt_encoder.named_parameters():
        if 'layer1' in name or 'layer2' in name or name=='conv1.weight' or name=='bn1.weight' or name=='bn1.bias':
          param.requires_grad = False

    if opt.model == 'loc_graph':
      self.frame_encoder = resnet_backbone()

    self.model = opt.model
    self.n_acts = opt.n_acts
    self.n_driver_acts = opt.n_driver_acts
    self.seq_len = opt.seq_len
    self.predict = opt.predict
    self.pred_seq_len = opt.pred_seq_len
    if self.predict:
      self.all_seq_len = self.seq_len + self.pred_seq_len
    else:
      self.all_seq_len = self.seq_len
    self.predict_k = opt.predict_k
    if self.predict_k:
      self.all_seq_len += 1

    # temporal modeling
    self.use_gru = opt.use_gru
    self.use_trn = opt.use_trn
    self.use_ped_gru = opt.ped_gru
    self.use_ctxt_gru = opt.ctxt_gru
    self.use_ctxt_node = opt.ctxt_node
    # loc graph
    self.use_frame_gru = opt.frame_gru
    self.use_node_gru = opt.node_gru

    self.conv_dim = opt.conv_dim
    self.branch = opt.branch
    self.pos_mode = opt.pos_mode
    self.use_act = opt.use_act
    self.use_gt_act = opt.use_gt_act
    self.use_driver = opt.use_driver
    self.use_pose = opt.use_pose
    self.use_signal = opt.use_signal
    # self.ped_dim = 2*self.conv_dim if self.branch == 'both' else self.conv_dim
    self.ped_dim = self.conv_dim


    if self.branch == 'ped' or self.branch == 'both':
      if self.pos_mode == 'center':
        self.ped_dim += 2
      elif self.pos_mode == 'height':
        self.ped_dim += 3
      elif self.pos_mode == 'bbox':
        self.ped_dim += 4
      elif self.pos_mode == 'both':
        self.ped_dim += 6
  
      if self.use_act:
        self.ped_dim += self.n_acts
  
      if self.use_pose and opt.model!='baseline_pose':
        self.ped_dim += 18 # x/y coor * 9 joints
  
      if self.use_signal:
        self.ped_dim += 2

    self.cls_dim = self.ped_dim + self.conv_dim if self.branch == 'both' else self.conv_dim

    if self.use_pose and self.model == 'graph':
      # Use pose in features of the pedestrian nodes.
      self.cls_dim -= 18 # since in model_graph, ped_feats are first mapped to of dimension conv_dim.

    if self.use_driver:
      # Use predicted driver behaviors in frame features.
      self.cls_dim += self.n_driver_acts
      self.driver_encoder = resnet_backbone()
      self.driver_gru = nn.GRU(self.conv_dim, self.conv_dim, 2, batch_first=True)
      self.driver_classifier = nn.Sequential(
       nn.Linear(self.conv_dim, 512),
       nn.ReLU(),
       nn.Linear(512, self.n_driver_acts),
      )
    
    # default: 2*512 (resnet conv) --> 2 (crossing or not)
    self.classifier = nn.Sequential(
       nn.Linear(self.cls_dim, 512),
       nn.ReLU(),
       nn.Linear(512, self.n_acts),
    )

    self.is_train = opt.is_train
    if self.is_train:
      self.lr_init = opt.lr_init
      self.lr_decay = opt.lr_decay
      self.decay_every = opt.decay_every
      self.wd = opt.wd
    self.extract_feats_dir = opt.extract_feats_dir

    self.optimizer = None
    self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
    if opt.reg_smooth == 'none':
      self.reg_fn = None
    elif opt.reg_smooth == 'l1':
      self.reg_fn = nn.L1Loss()
    elif opt.reg_smooth == 'l2':
      self.reg_fn = nn.MSELoss()
    elif opt.reg_smooth == 'hinge':
      self.reg_fn = nn.HingeEmbeddingLoss(margin=0.2)
    self.reg_lambda = opt.reg_lambda

    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()


  def setup(self):
    if not self.is_train:
      return

    self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_init, weight_decay=self.wd, betas=(0.5, 0.999))
    self.train()


  def update_hyperparameters(self, epoch):
    '''
    Update hyperparameters, including learning rates, lambdas, etc.
    Hard-code rules for now.
    '''
    # Learning rate
    lr = self.lr_init
    if self.lr_decay:
      if (epoch+1) % self.decay_every == 0:
        lr = self.lr_init * (0.1**((epoch+1) // self.decay_every))
      for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr


  def step_train(self, data):
    ped_crops, masks = data['ped_crops'], data['all_masks']
    act = data['GT_act']
    act = act.type(self.dtype).to(self.device)
    obj_bbox, obj_cls = data['obj_bbox'], data['obj_cls']
    if 'GT_ped_bbox' in data:
      # loc-centric
      bboxes = data['GT_ped_bbox']
    else:
      # ped-centric
      bboxes, pose = data['GT_bbox'], None
      if 'GT_pose' in data:
        pose = data['GT_pose']
        pose = pose.to(self.device)

    frames, driver_act = None, None
    if 'frames' in data:
      frames = data['frames']
    if self.use_driver:
      driver_act = data['GT_driver_act']
      driver_act = driver_act.type(self.dtype).to(self.device)

    if self.model == 'graph':
      ret = self.forward(ped_crops, masks, ped_bbox=bboxes, act=act, pose=pose, obj_bbox=obj_bbox, obj_cls=obj_cls, frames=frames)
    elif 'loc' not in self.model:
      ret = self.forward(ped_crops, masks, bboxes, act=act, pose=act)
    else:
      # loc-centric models
      ret = self.forward(ped_crops, masks, bboxes, act=act, obj_bbox=obj_bbox, obj_cls=obj_cls, frames=frames)

    if type(ret) == dict:
      logits = ret['logits']
      act_logits = ret['act_logits'] if 'act_logits' in ret else None
      driver_act_logits = ret['driver_act_logits'] if 'driver_act_logits' in ret else None
    elif type(ret)==tuple and len(ret) == 2:
      logits, act_logits = ret
      driver_act_logits = None
    else:
      logits = ret
      act_logits, driver_act_logits = None, None

    if act.shape[1] > logits.shape[1]:
      if logits.shape[1] != self.pred_seq_len:
        raise ValueError("logits should be for prediction. Expecting pred_seq_len={}, got {}".format(
          self.pred_seq_len, logits.shape[1]))
      act = act[:, self.seq_len:]

    if self.n_acts == 1:
      if act.shape[-1] > 1:
        crossing = act[..., 1]
      else:
        # act only contains crossing labels
        crossing = act[..., 0]
      loss = self.loss_fn(logits.view(-1), crossing.view(-1))
      if act_logits is not None:
        loss += self.loss_fn(act_logits.view(-1), crossing[:,:self.seq_len].view(-1))
    else:
      loss = self.loss_fn(logits, act)
      if act_logits is not None:
        loss += self.loss_fn(act_logits, act[:, :self.seq_len])

    if driver_act_logits is not None:
      loss += self.loss_fn(driver_act_logits, driver_act[:, :self.seq_len])

    if self.reg_fn is not None:
      loss += self.reg_lambda * self.reg_fn(logits[:, 1:], logits[:, :-1])

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()

 
  def step_test(self, data, slide=True, collect_A=False):
    ped_crops, masks = data['ped_crops'], data['all_masks']
    act = data['GT_act']
    act = act.type(self.dtype).to(self.device)
    obj_bbox, obj_cls = data['obj_bbox'], data['obj_cls']
    if 'GT_ped_bbox' in data:
      # loc-centric
      bboxes = data['GT_ped_bbox']
      B = len(ped_crops)
      T = len(ped_crops[0])
    else:
      # ped-centric
      bboxes, pose = data['GT_bbox'], None
      if 'GT_pose' in data:
        pose = data['GT_pose']
        pose = pose.to(self.device)

      if ped_crops.dim() == 5:
        B, T, _, _, _ = ped_crops.shape # (B, T, 3, 224, 224)
      else:
        B = ped_crops.shape[0]
        T = ped_crops.shape[1]

    frames = data['frames'] if 'frames' in data else None

    As = None

    if not slide:
      if T>self.all_seq_len:
        # randomly sample a snipped to test
        t_start = random.randint(0, T-sell.all_seq_len)
      else:
        t_start = 0
      if type(ped_crops) == list:
        ped_crops = [each[t_start:t_start+self.seq_len] for each in ped_crops]
      else:
        ped_crops = ped_crops[:, t_start:t_start+self.seq_len]
      masks = [each[t_start:t_start+self.seq_len] for each in masks]
      act = act[:, t_start:t_start+self.all_seq_len]
      if self.model == 'graph':
        logits = self.forward(ped_crops, masks, bboxes, act=act, pose=pose,
                              obj_bbox=obj_bbox, obj_cls=obj_cls, frames=frames, collect_A=collect_A)
      elif 'loc' not in self.model:
        logits = self.forward(ped_crops, masks, bboxes, act=act, pose=pose)
      else:
        # loc-centric models
        logits = self.forward(ped_crops, masks, bboxes, act=act, obj_bbox=obj_bbox, obj_cls=obj_cls, frames=frames)

      if type(logits) == dict:
        if 'As' in logits:
          As = logits['As']
        logits = logits['logits']
      elif type(logits) == tuple and len(logits) == 2:
        logits = logits[0]

    else: # i.e. if slide
      logits = [[] for _ in range(T)]
      for t_start in range(T):
        curr_ped_crops = ped_crops[:, t_start:t_start+self.seq_len]
        curr_masks = [each[t_start:t_start+self.seq_len] for each in masks]
        curr_acts = act[:, t_start:t_start+self.all_seq_len]
        if self.model == 'graph':
          curr_logits = self.forward(ped_crops, masks, ped_bbox=bboxes, act=act, pose=pose, 
                                     obj_bbox=obj_bbox, obj_cls=obj_cls, frames=frames, collect_A=collect_A)
        elif 'loc' not in self.model:
          curr_logits = self.forward(ped_crops, masks, bbox=bboxes, act=act, pose=pose)
        else:
          # loc-centric models
          curr_logits = self.forward(ped_crops, masks, bbox=bboxes, act=act, obj_bbox=obj_bbox, obj_cls=obj_cls, frames=frames)

        if type(curr_logits) == dict:
          if 'As' in logits:
            As = curr_logits['As']
          curr_logits = curr_logits['logits']
        elif type(curr_logits) == tuple and len(curr_logits) == 2:
          curr_logits = curr_logits[0]

        for delta in range(curr_logits.shape[1]):
          logits[t_start+delta] += curr_logits[:, delta],
      for t in range(T):
        logits[t] = torch.cat(logits[t], -1).mean(-1)
      logits = torch.cat(logits, -1)

    if act.shape[-1] > 1:
      crossing = act[:, :, 1]
    else:
      # act only contains crossing or not
      crossing = act[:, :, 0]

    if self.predict or self.predict_k:
      if logits.shape[1] == self.all_seq_len:
        logits_det, crossing_det = logits[:, :self.seq_len], crossing[:, :self.seq_len]
        det_metrics = self.compute_metrics(logits_det, crossing_det)
      else:
        # e.g. baseline_antipate_cnn: no detection part
        det_metrics = None

      logits_pred, crossing_pred = logits[:, self.seq_len:], crossing[:, self.seq_len:]
      pred_metrics = self.compute_metrics(logits_pred, crossing_pred)

      # compute metrics for the farthest frame
      logits_last, crossing_last = logits_pred[:, -1:], crossing[:, -1:]
      last_metrics = self.compute_metrics(logits_last, crossing_last)

      return det_metrics, pred_metrics, last_metrics, As
    else:
      det_metrics = self.compute_metrics(logits, crossing)
      return det_metrics, None, None, As


  def compute_metrics(self, logits, crossing):
    # logits: shape: (B, T) or (B, T, n_acts)
    # crossing: shape: (B, T)
    B = logits.shape[0]
    # probabilities of predicting 1
    if logits.dim() == 3 and logits.shape[-1]>1:
      logits = logits[:, :, 1]
    probs = self.sigmoid(logits).view(B, -1)

    # probabilities of predicting the true label
    true_probs = probs * (crossing == 1).type(self.dtype) + (1-probs) * (crossing == 0).type(self.dtype)
    # tmp_preds = (true_probs > 0.5).type(self.dtype)
    # tmp_n_correct_frames = (tmp_preds == 1).type(self.dtype).sum()

    preds = (probs > 0.55).type(self.dtype).view(B, -1)

    # preds2 = tmp_preds * (crossing == 1).type(self.dtype) + (1 - tmp_preds) * (crossing == 0).type(self.dtype)
    # preds2 = preds2.type(self.dtype).view(b, -1)
    # tmp_n_correct_frames = (preds2 == crossing).type(self.dtype).sum()


    # frame-level
    n_correct_frames = (preds == crossing).type(self.dtype).sum()
    loss = self.loss_fn(logits.view(-1), crossing.view(-1))
    n_frames = logits.numel()

    # clip-level
    n_correct_clips = (preds.max(1)[0] == crossing.max(1)[0]).sum()

    # by class
    n_cross = crossing.sum().item()
    n_non_cross = crossing.numel() - n_cross
    n_correct_cross = ((crossing == 1) * (preds == crossing)).sum().item()
    n_correct_non_cross = ((crossing == 0) * (preds == crossing)).sum().item()

    # pdb.set_trace()

    return n_frames, n_correct_frames.item(), B, n_correct_clips.item(), \
      n_cross, n_non_cross, n_correct_cross, n_correct_non_cross, \
      true_probs.sum(0).cpu().detach(), loss.item() * n_frames, \
      preds.cpu().detach(), crossing.cpu().detach()

  def append_pos(self, feats, bbox):
    bbox = bbox.to(self.device)
    if self.pos_mode in ['center', 'both', 'height']:
      y = bbox[:,:, 0] + .5*bbox[:,:,2]
      x = bbox[:,:, 1] + .5*bbox[:,:,3]
      y = y.unsqueeze(-1)
      x = x.unsqueeze(-1)
      centers = torch.cat([y,x], -1)
      if self.pos_mode == 'both':
        feats_pos = torch.cat([feats, centers, bbox], -1)
      elif self.pos_mode == 'height':
        h = bbox[:,:,2:3]
        feats_pos = torch.cat([feats, centers, h], -1)
      else:
        feats_pos = torch.cat([feats, centers], -1)
    elif self.pos_mode == 'bbox':
      feats_pos = torch.cat([feats, bbox], -1)

    return feats_pos

  def util_norm_pose(self, pose):
    # Normalize pose

    # take det part only + discard confidence (i.e. last col)
    x = pose[:, :self.seq_len, :, :2]

    neck = x[:, :, 0]
    rhip = x[:, :, 3]
    lhip = x[:, :, 6]
    # center = (LHip + RHip) / 2
    center = (rhip + lhip) / 2
    # torso height = neck[y] - (LHip[y] + RHip[y])/2
    th = neck[:, :, 1] - (rhip[:,:,1] + lhip[:,:,1])/2
    th[th==0] = 1

    # subtract center
    x -= center.unsqueeze(-2)
    # scale by torch height
    x /= th.unsqueeze(-1).unsqueeze(-1)

    return x


  def save(self, ckpt_path, epoch, name=''):
    out = {
      'model': self.state_dict(),
      'optim': self.optimizer.state_dict(),
      'epoch': epoch,
    }
    if name:
      torch.save(out, os.path.join(ckpt_path, '{}.pth'.format(name)))
    else:
      torch.save(out, os.path.join(ckpt_path, '{}.pth'.format(epoch)))


  def load(self, ckpt_path):
    ckpt = torch.load(ckpt_path)

    # load model
    updated_params = {}
    new_params = self.state_dict()
    new_keys = list(new_params.keys())
    for k,v in ckpt['model'].items():
      if k in new_keys:
        updated_params[k] = v
    new_params.update(updated_params)
    self.load_state_dict(new_params)

    if self.optimizer is not None:
      # load optim
      updated_params = {}
      new_params = self.optimizer.state_dict()
      new_keys = list(new_params.keys())
      for k,v in ckpt['optim'].items():
        if k in new_keys:
          updated_params[k] = v
      new_params.update(updated_params)
      self.optimizer.load_state_dict(new_params)
