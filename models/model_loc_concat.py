import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from models.backbone.resnet_based import resnet_backbone

from models.model_base import BaseModel

import pdb


class LocConcatModel(BaseModel):
  def __init__(self, opt):
    super().__init__(opt)
    if self.use_gru:
      self.gru = nn.GRU(self.cls_dim, self.cls_dim, 2, batch_first=True).to(self.device)
    elif self.use_trn:
      if self.predict and self.pred_seq_len != 1:
        raise ValueError("self.pred_seq_len has to be 1 when using TRN.")
      self.k = 3 # number of samples per branch
      self.f2 = nn.Sequential(
        nn.Linear(2*self.cls_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256))
      self.f3 = nn.Sequential(
        nn.Linear(3*self.cls_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256))
      self.f4 = nn.Sequential(
        nn.Linear(4*self.cls_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256))
      self.h2 = nn.Linear(256*self.k, self.n_acts)
      self.h3 = nn.Linear(256*self.k, self.n_acts)
      self.h4 = nn.Linear(256*self.k, self.n_acts)

    if self.use_ped_gru:
      self.ped_gru = nn.GRU(self.ped_dim, self.ped_dim, 2, batch_first=True).to(self.device)
    if self.use_ctxt_gru:
      self.ctxt_gru = nn.GRU(self.conv_dim, self.conv_dim, 2, batch_first=True).to(self.device)

    if self.use_act and not self.use_gt_act:
      self.act_gru = nn.GRU(self.cls_dim-self.n_acts, self.cls_dim, 2, batch_first=True).to(self.device)
      

    if self.predict:
      # NOTE: this GRU only takes in seq of len 1, i.e. one time step
      # GRU is used over GRUCell for multilayer
      self.gru_pred = nn.GRU(self.cls_dim, self.cls_dim, 2, batch_first=True).to(self.device)

    if self.predict_k:
      self.fc_pred = nn.Sequential(
        nn.Linear(self.cls_dim, 256),
        nn.ReLU(),
        nn.Linear(256, self.cls_dim)
      )


  def forward(self, ped_crops, masks, bbox=None, act=None, obj_bbox=None, obj_cls=None):
    B = len(ped_crops)
    T = len(ped_crops[0])

    if self.branch in ['both', 'ped']:
      # ped_crops: (bt, seq_len, [n_peds, (3, 224, 224)]
      ped_feats = [[] for _ in range(B)]
      for b in range(B):
        for t in range(T):
          if len(ped_crops[b][t]) == 0:
            ped_feat = torch.zeros([self.conv_dim]).to(self.device)
          else:
            curr_ped_crops = torch.tensor(ped_crops[b][t]).type(self.dtype).view(-1, 3, 224, 224)
            curr_ped_crops = curr_ped_crops.to(self.device)
            ped_feat = self.ped_encoder(curr_ped_crops)
            ped_feat = ped_feat.view(-1, self.conv_dim).mean(0)
          ped_feats[b] += ped_feat,
        ped_feats[b] = torch.stack(ped_feats[b])
      # ped_feats: (B, T, d)
      ped_feats = torch.stack(ped_feats)
      if self.use_ped_gru:
        ped_feats, _ = self.ped_gru(ped_feats)

    if self.branch in ['both', 'ctxt']:
      # masks: list: (bt, 30, {cls: (n, 3, 224, 224)})
      ctxt_feats = [[] for _ in range(B)]
      for b in range(B):
        for t in range(T):
          if len(masks[b][t]) == 0:
            ctxt_feat = torch.zeros([self.conv_dim])
            ctxt_feat = ctxt_feat.to(self.device)
          else:
            if type(masks[b][t]) is dict:
              vals = list(masks[b][t].values())
              # ctxt_masks: (n_total, (3,) 224, 224)
              ctxt_masks = torch.cat(vals, 0)
            else:
              ctxt_masks = masks[b][t]
            # TODO: this seems to be a bug though it didn't complain before.
            # Check whether this will affect the prev model.
            # ctxt_masks = ctxt_masks.sum(0, True)
            if ctxt_masks.dim() == 3:
              n_total, h, w = ctxt_masks.shape
              ctxt_masks = ctxt_masks.unsqueeze(1).expand([n_total, 3, h, w])
            elif ctxt_masks.dim() == 2:
              h, w = ctxt_masks.shape
              ctxt_masks = ctxt_masks.unsqueeze(0).unsqueeze(0).expand([1, 3, h, w])
            ctxt_masks = ctxt_masks.to(self.device)
            # ctxt_feats: (n_total, d)
            # print('ctxt_masks', ctxt_masks.shape)
            ctxt_feat = self.ctxt_encoder(ctxt_masks.type(self.dtype))
            # average pool
            ctxt_feat = ctxt_feat.mean(0).squeeze(-1).squeeze(-1)
          ctxt_feats[b] += ctxt_feat,
        ctxt_feats[b] = torch.stack(ctxt_feats[b])
      ctxt_feats = torch.stack(ctxt_feats)


    if os.path.exists(self.extract_feats_dir) and False:
      # NOTE: set self.rand_test = 0 when extracting features since we want to cover all frames
      feat_path = os.path.join(self.extract_feats_dir, 'ped{}.pkl'.format(idx))
      with open(feat_path, 'wb') as handle:
        feats = {
          'ped_feats': ped_feats,
          'ctxt_feats': ctxt_feats,
        }
        pickle.dump(feats, handle)

    if self.pos_mode != 'none':
      ped_feats = self.append_pos(ped_feats, bbox[:, :self.seq_len])

    if self.use_signal:
      # act 2: handwave / act 3: looking
      signal = act[:, :, 2:4].to(self.device)
      ped_feats = torch.cat([ped_feats, signal], -1)
    
    if self.branch == 'both':
      frame_feats = torch.cat([ped_feats, ctxt_feats], -1)
    elif self.branch == 'ped':
      frame_feats = ped_feats
    elif self.branch == 'ctxt':
      frame_feats = ctxt_feats
    else:
      raise ValueError("self.branch should be 'both', 'ped', or 'ctxt'. Got {}".format(self.branch))

    if self.use_act:
      if self.use_gt_act:
        if self.n_acts == 1:
          act = act[:, :, 1:2]
        act = act[:, :self.seq_len]
      else:
        # use predicted action labels
        temporal_feats, _ = self.act_gru(frame_feats)
        h = self.classifier(temporal_feats)
        act_logits = self.sigmoid(h)
        act = (act_logits > 0.5).type(self.dtype)
      frame_feats = torch.cat([frame_feats, act], -1)

    if self.use_pose:
      normed_pose = self.util_norm_pose(pose)
      normed_pose = normed_pose.contiguous().view(B, T, -1)
      frame_feats = torch.cat([frame_feats, normed_pose], -1)

    if self.use_gru:
      # self.gru keeps the dimension of frame_feats
      frame_feats, h = self.gru(frame_feats)
    elif self.use_trn:
      # Note: for predicting the next frame only (i.e. Lopez's setting)
      feats2 = []
      feats3 = []
      feats4 = []
      for _ in range(self.k):
        # 2-frame relations
        l2 = self.seq_len // 2
        id1 = random.randint(0, l2-1)
        id2 = random.randint(0, l2-1) + l2
        feat2 = torch.cat([frame_feats[:, id1], frame_feats[:, id2]], -1)
        feats2 += self.f2(feat2),
        # 3-frame relations
        l3 = self.seq_len // 3
        id1 = random.randint(0, l3-1)
        id2 = random.randint(l3, 2*l3-1)
        id3 = random.randint(2*l3, self.seq_len-1)
        feat3 = torch.cat([frame_feats[:, id1], frame_feats[:, id2], frame_feats[:, id3]], -1)
        feats3 += self.f3(feat3),
        # 4-frame relations
        l4 = self.seq_len // 4
        id1 = random.randint(0, l4-1)
        id2 = random.randint(l4, 2*l4-1)
        id3 = random.randint(2*l4, 3*l4-1)
        id4 = random.randint(3*l4, self.seq_len-1)
        feat4 = torch.cat([frame_feats[:, id1], frame_feats[:, id2], frame_feats[:, id3], frame_feats[:, id4]], -1)
        feats4 += self.f4(feat4),
      t2 = self.h2(torch.cat(feats2, -1))
      t3 = self.h2(torch.cat(feats3, -1))
      t4 = self.h2(torch.cat(feats4, -1))
      logits = t2 + t3 + t4
      logits = logits.view(B, 1, self.n_acts)
      return logits

    if self.predict:
      # o: (B, 1, cls_dim)
      o = frame_feats[:, -1:]
      pred_outs = []
      for pred_t in range(self.pred_seq_len):
        o, h = self.gru_pred(o, h)
        # if self.pos_mode != 'none':
        #   o = self.append_pos(o, bbox[:, self.seq_len+pred_t:self.seq_len+pred_t+1])
        pred_outs += o,

      # pred_outs: (B, pred_seq_len, cls_dim)
      pred_outs = torch.cat(pred_outs, 1)
      # frame_feats: (B, seq_len + pred_seq_len, cls_dim)
      frame_feats = torch.cat([frame_feats, pred_outs], 1)

    if self.predict_k:
      # h: (n_gru_layers, B, cls_dim) --> (B, 1, cls_dim)
      h = h.transpose(0,1)[:, -1:]
      # pred_feats: (B, T, cls_dim)
      pred_feats = self.fc_pred(h)
      frame_feats = torch.cat([frame_feats, pred_feats], 1)

    # shape: (B, T, 2)
    logits = self.classifier(frame_feats)
    if self.use_act and not self.use_gt_act:
      return logits, act_logits
    #   logits[:, :self.seq_len] = act_logits

    return logits


  def extract_feats(self, ped_crops, masks):
    """
    ped_crops: a list 
    masks: list of len 1; each item being a dict w/ key 1-4
    """
    assert(len(masks) == 1)
    B = len(ped_crops)
    T = len(ped_crops[0])

    ped_feats = [[] for _ in range(B)]
    for b in range(B):
      for t in range(T):
        if len(ped_crops[b][t]) == 0:
          ped_feat = torch.zeros([self.conv_dim]).to(self.device)
        else:
          curr_ped_crops = torch.tensor(ped_crops[b][t]).type(self.dtype).view(-1, 3, 224, 224)
          curr_ped_crops = curr_ped_crops.to(self.device)
          ped_feat = self.ped_encoder(curr_ped_crops)
          ped_feat = ped_feat.view(-1, self.conv_dim).mean(0)
        ped_feats[b] += ped_feat,
      ped_feats[b] = torch.stack(ped_feats[b])

    ctxt_feats = []
    ctxt_cls = []
    for t,mask in enumerate(masks):
      if len(mask) == 0:
        ctxt_feat = torch.zeros([1, self.conv_dim])
        ctxt_cls += [0]
      else:
        if type(mask) is dict:
          # grouped by classes
          ctxt_masks = []
          for k in sorted(mask):
            for each in mask[k]:
              ctxt_cls += k,
              ctxt_masks += each,
          ctxt_masks = torch.stack(ctxt_masks, 0)
        else:
          # class collapsed
          ctxt_masks = mask
          ctxt_cls += [0] * ctxt_masks.shape[0]

        if ctxt_masks.dim() == 3:
          n_total, h, w = ctxt_masks.shape
          ctxt_masks = ctxt_masks.unsqueeze(1).expand([n_total, 3, h, w])
        ctxt_masks = ctxt_masks.to(self.device)

        # ctxt_feats: (n_total, d)
        ctxt_feat = self.ctxt_encoder(ctxt_masks.type(self.dtype)).cpu()
        ctxt_feat = ctxt_feat.squeeze(-1).squeeze(-1)
      ctxt_feats += ctxt_feat,
    ctxt_feats = torch.cat(ctxt_feats, 0)
    assert(len(ctxt_feats) == len(ctxt_cls))
    
    # NOTE: set self.rand_test = 0 when extracting features since we want to cover all frames
    return ped_feats, ctxt_feats, ctxt_cls

