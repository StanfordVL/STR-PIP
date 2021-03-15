import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.resnet_based import resnet_backbone

from models.model_base import BaseModel

import pdb


class PosModel(BaseModel):
  def __init__(self, opt):
    super().__init__(opt)
    self.seq_len = opt.seq_len
    self.pos_mode = opt.pos_mode
    if self.pos_mode == 'center':
      self.fc_in_dim = 2
    elif self.pos_mode == 'height':
      self.fc_in_dim = 3
    elif self.pos_mode == 'bbox':
      self.fc_in_dim = 4
    elif self.pos_mode == 'both':
      self.fc_in_dim = 6
    elif self.pos_mode == 'none':
      self.fc_in_dim = 0

    self.use_gt_act = opt.use_gt_act
    if self.use_gt_act:
      self.fc_in_dim += self.n_acts

    if self.fc_in_dim == 0:
      raise ValueError("The model should use at least one of 'pos_mode' or 'use_gt_act'.")
    
    self.gru = nn.GRU(self.fc_in_dim, self.fc_in_dim, 2, batch_first=True).to(self.device)
    self.classifier = nn.Sequential(
        nn.Linear(self.fc_in_dim, self.fc_in_dim),
        nn.Linear(self.fc_in_dim, self.n_acts)
    )

    if self.predict:
      # NOTE: this GRU only takes in seq of len 1, i.e. one time step
      # GRU is used over GRUCell for multilayer
      self.gru_pred = nn.GRU(self.fc_in_dim, self.fc_in_dim, 2, batch_first=True).to(self.device)
    
  
  def forward(self, ped, ctxt, bbox, act, pose=None):
    # bbox: (B, T, 4): (y, x, h, w)
    bbox = bbox.to(self.device)
    if self.pos_mode in ['center', 'both', 'height']:
      y = bbox[:,:, 0] + .5*bbox[:,:,2]
      x = bbox[:,:, 1] + .5*bbox[:,:,3]
      y = y.unsqueeze(-1)
      x = x.unsqueeze(-1)
      centers = torch.cat([y,x], -1)
      if self.pos_mode == 'both':
        gru_in = torch.cat([centers, bbox], -1)
      elif self.pos_mode == 'height':
        h = bbox[:,:,2:3]
        gru_in = torch.cat([centers, h], -1)
      else:
        gru_in = centers
    elif self.pos_mode == 'bbox':
      gru_in = bbox
    elif self.pos_mode == 'none':
      gru_in = None
    
    if self.use_gt_act:
      # shape: (bt, all_seq_len, n_acts+len_for_pos)
      if self.n_acts == 1:
        act = act[:, :, 1:2]
      if gru_in is None:
        gru_in = act
      else:
        gru_in = torch.cat([gru_in, act], -1)

    if self.predict:
      gru_in = gru_in[:, :self.seq_len]

    output, h = self.gru(gru_in)

    if self.predict:
      # o: (B, 1, fc_in_dim)
      o = output[:, -1:]
      pred_outs = []
      for _ in range(self.pred_seq_len):
        o, h = self.gru_pred(o, h)
        pred_outs += o,

      # pred_outs: (B, pred_seq_len, fc_in_dim)
      pred_outs = torch.cat(pred_outs, 1)
      # frame_feats: (B, seq_len + pred_seq_len, fc_in_dim)
      output = torch.cat([output, pred_outs], 1)

    logits = self.classifier(output)
    return logits
