import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_base import BaseModel

import pdb

class BaselineAnticipateCNN(BaseModel):
  def __init__(self, opt):
    super().__init__(opt)
    # nRows: seq_len
    # nCols: n_acts, i.e. acts.shape[1]
    self.W1 = nn.Conv2d(1, 8, kernel_size=(5,1), padding=(2,0))
    self.W2 = nn.Conv2d(8, 16, kernel_size=(5,1), padding=(2,0))
    self.fc1 = nn.Linear(16*self.seq_len*(self.n_acts//4), 1024)
    self.fc2 = nn.Linear(1024, self.seq_len*self.n_acts)

    # init params
    self.W1.weight.data.normal_(std=0.1)
    self.W1.bias.data.fill_(0.1)
    self.W2.weight.data.normal_(std=0.1)
    self.W2.bias.data.fill_(0.1)

    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
    self.l2_norm = F.normalize

    if not self.use_gt_act:
      self.gru = nn.GRU(self.ped_dim, self.ped_dim, 2, batch_first=True).to(self.device)

  def forward(self, ped, masks, bbox, act, pose=None):
    if self.use_gt_act:
      # use GT action observations
      x = act[:, :self.seq_len].unsqueeze(1)
    else:
      # predict action labels from pedestrian crops

      # ped_crops: (bt, 30, 3, 224, 224)
      B, T, _, _, _ = ped.shape

      ped_crops = ped.view(-1, 3, 224, 224)
      ped_crops = ped_crops.type(self.dtype).to(self.device)
      ped_feats = self.ped_encoder(ped_crops)
      # ped_feats: (B, T, d)
      ped_feats = ped_feats.view(B, T, -1)

      temporal_feats, _ = self.gru(ped_feats)
      h = self.classifier(temporal_feats)
      logits = self.sigmoid(h)
      x = (logits > 0.5).type(self.dtype)
      x = x.unsqueeze(1)

    # x = acts[:, :self.seq_len].unsqueeze(1)
    x = self.W1(x)
    x = self.relu(x)
    # max pool over the channel dimension
    x = self.pool(x)

    x = self.W2(x)
    x = self.relu(x)
    # max pool over the channel dimension
    x = self.pool(x)

    x = x.view(-1, 16*self.seq_len*(self.n_acts // 4))
    x = self.fc1(x)
    x = self.fc2(x)

    x = x.view(-1, self.seq_len, self.n_acts)
    x = self.l2_norm(x, dim=2)

    if not self.use_gt_act:
      # also supervise on the detection
      x = torch.cat([h, x], 1)

    return x
