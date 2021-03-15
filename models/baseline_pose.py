import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_base import BaseModel

import pdb

class BaselinePose(BaseModel):
  def __init__(self, opt):
    super().__init__(opt)

    if self.use_gru:
      # aggregate poses with FC + GRU
      self.embed_pose = nn.Sequential(
        nn.Linear(9*2, 128),
        nn.ReLU(),
        nn.Linear(128, self.fc_in_dim),
      )
      self.gru = nn.GRU(self.fc_in_dim, self.fc_in_dim, 2, batch_first=True)
      if self.predict:
        self.gru_pred = nn.GRU(self.fc_in_dim, self.fc_in_dim, 2, batch_first=True)
    else:
      # aggregating poses using 1D Conv as in BaselineAnticipateCNN
      # nRows: seq_len
      # nCols: n_acts, i.e. acts.shape[1]
      self.W1 = nn.Conv2d(1, 8, kernel_size=(5,1), padding=(2,0))
      self.W2 = nn.Conv2d(8, 16, kernel_size=(5,1), padding=(2,0))
      hidden_size = 256
      self.fc1 = nn.Linear(16*self.seq_len*((9*2)//4), hidden_size) # 9 joints x 2 coor (x & y)
      self.fc2 = nn.Linear(hidden_size, self.pred_seq_len*self.n_acts)

      # init params
      self.W1.weight.data.normal_(std=0.1)
      self.W1.bias.data.fill_(0.1)
      self.W2.weight.data.normal_(std=0.1)
      self.W2.bias.data.fill_(0.1)

      self.relu = nn.ReLU()
      self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
      self.l2_norm = F.normalize

      # if not self.use_gt_act:
      #   self.gru = nn.GRU(self.fc_in_dim, self.fc_in_dim, 2, batch_first=True).to(self.device)

  def forward(self, ped, masks, bbox, act, pose=None):
    # use GT action observations
    B = ped.shape[0]

    x = self.util_norm_pose(pose)
    
    x = x.contiguous().view(B, self.seq_len, 18)

    if self.use_gru:
      feats = self.embed_pose(x)
      temp_feats, h = self.gru(feats)

      if self.predict:
        o = temp_feats[:, -1:]
        pred_outs = []
        for pred_t in range(self.pred_seq_len):
          o, h = self.gru_pred(o, h)
          pred_outs += o,
        pred_outs = torch.cat(pred_outs, 1)
        temp_feats = torch.cat([temp_feats, pred_outs], 1)

      logits = self.classifier(temp_feats)
    else:
      x = x.unsqueeze(1) # shape: (B, 1, self.seq_len, 18)
      x = self.W1(x)
      x = self.relu(x)
      # max pool over the channel dimension
      x = self.pool(x)
  
      x = self.W2(x)
      x = self.relu(x)
      # max pool over the channel dimension
      x = self.pool(x)
  
      x = x.view(-1, 16*self.seq_len*((9*2) // 4))
      x = self.fc1(x)
      x = self.fc2(x)
  
      x = x.view(-1, self.pred_seq_len, self.n_acts)
      logits = self.l2_norm(x, dim=2)

    # if not self.pred_only:
    #   x = torch.cat([h, x], 1)
    return logits
