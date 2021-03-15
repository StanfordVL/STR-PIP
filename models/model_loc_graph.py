import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.backbone.resnet_based import resnet_backbone

from models.model_base import BaseModel

from itertools import accumulate
import pdb


class LocGraphModel(BaseModel):
  def __init__(self, opt):
    super().__init__(opt)
    if self.use_gru:
      self.gru = nn.GRU(self.cls_dim, self.cls_dim, 2, batch_first=True).to(self.device)
    if self.use_frame_gru:
      self.frame_gru = nn.GRU(self.conv_dim, self.conv_dim, 2, batch_first=True).to(self.device)
    if self.use_node_gru:
      self.node_gru = nn.GRU(self.conv_dim, self.conv_dim, 2, batch_first=True).to(self.device)

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

    self.n_layers = opt.n_layers

    # adjacency matrix
    self.adj_type = opt.adj_type
    if self.adj_type == 'inner' or self.adj_type == 'cosine':
      self.frame_embed = nn.Linear(self.conv_dim, 512)
      self.node_embed = nn.Linear(self.conv_dim, 512)
      self.sm = nn.Softmax(-1)
    elif self.adj_type == 'spatial':
      self.frame_embed = nn.Linear(self.conv_dim+64, 128)
      self.node_embed = nn.Linear(self.conv_dim, 128)
      self.pos_embed = nn.Linear(8, 64)
    elif self.adj_type == 'prior':
      # TODO: save cls info (prior is defined over classes)
      raise NotImplementedError("'Prior' not implemented.")

    # layer weight
    self.W = nn.Linear(self.conv_dim, self.conv_dim)

  def backbone_feats(self, ped_crops, masks):
    B, T, _,_,_ = ped_crops.shape

    # ped_crops: (bt, 30, 3, 224, 224)
    ped_crops = ped_crops.view(-1, 3, 224, 224)
    ped_crops = ped_crops.type(self.dtype)
    ped_crops = ped_crops.to(self.device)
    ped_feats = self.ped_encoder(ped_crops)
    # ped_feats: (B, T, d)
    ped_feats = ped_feats.view(B, T, -1)

    # masks: list: (bt, 30, {cls: (n, 3, 224, 224)})
    ctxt_feats = [[] for _ in range(B)]
    for b in range(B):
      for t in range(T):
        if type(masks[b][t]) == dict:
          vals = list(masks[b][t].values())
          if len(vals):
            ctxt_masks = torch.cat(vals, 0)
        else:
          vals = masks[b][t]
          ctxt_masks = vals
        if len(vals) == 0:
          ctxt_feat = torch.zeros([self.conv_dim])
          ctxt_feat = ctxt_feat.to(self.device)
        else:
          # ctxt_masks: (n_total, 3, 224, 224)

          if ctxt_masks.dim() == 3:
            n_total, h, w = ctxt_masks.shape
            ctxt_masks = ctxt_masks.unsqueeze(1).expand([n_total, 3, h, w])
          ctxt_masks = ctxt_masks.to(self.device)
          # ctxt_feats: (n_total, d)
          ctxt_feat = self.ctxt_encoder(ctxt_masks.type(self.dtype))
          ctxt_feat = ctxt_feat.squeeze(-1).squeeze(-1) # shape: [n_objs, 512]
        ctxt_feats[b] += ctxt_feat,

    return ped_feats, ctxt_feats


  def getA(self, n_objs, frame_enc=None, curr_nodes=None, curr_ped=None, curr_objs=None, ped_bbox=None, obj_bbox=None):
    # curr_ped: (1, D)
    # curr_objs: (n_objs, D)
    A = torch.diag(torch.ones([n_objs+1]))
    if self.adj_type == 'uniform':
      A[1:,0] = 1/n_objs
      A[0, 1:] = 1/n_objs
    elif self.adj_type == 'random':
      ws = torch.rand([n_objs])
      if ws.sum() != 0:
        ws = ws / ws.sum()
      A[1:, 0] = ws
      A[0, 1:] = ws
    elif self.adj_type == 'inner' or self.adj_type == 'cosine':
      embed_frame = self.frame_embed(frame_enc)
      embed_nodes = self.node_embed(curr_nodes)
      if self.adj_type == 'inner':
        dot = torch.matmul(embed_frame, embed_nodes.transpose(0,1))
        ws = self.sigmoid(dot)
        # normed_dot = (dot - dot.mean()) / (dot.max()-dot.min())
        # ws = self.sm(dot)
      elif self.adj_type == 'cosine':
        ws = F.cosine_similarity(embed_frame, embed_nodes)
      A[1:, 0] = ws
      A[0, 1:] = ws
    elif self.adj_type == 'spatial':
      raise NotImplementedError("Spatial not implemented for Loc Graph.")
      # calculating rel positions
      # ped_bbox: (1, 4): (x_min, y_min, w, h)
      # obj_bbox: (n_objs, 4)
      obj_bbox = torch.tensor(obj_bbox).type(self.dtype).to(self.device)
      if obj_bbox.dim() == 1:
        # i.e. obj_bbox = tensor([])
        ws = torch.tensor([0])
      else:
        e_ped_bbox = ped_bbox.repeat([n_objs, 1])
        dxmin = abs(e_ped_bbox[:, 0] - obj_bbox[:, 0])
        dymin = abs(e_ped_bbox[:, 1] - obj_bbox[:, 1])
        dxmax = abs(e_ped_bbox[:, 0]+e_ped_bbox[:, 2] - obj_bbox[:, 0]-obj_bbox[:, 2])
        dymax = abs(e_ped_bbox[:, 1]+e_ped_bbox[:, 3] - obj_bbox[:, 1]-obj_bbox[:, 3])
        dxc = abs(e_ped_bbox[:, 0]+0.5*e_ped_bbox[:, 2] - obj_bbox[:, 0]-0.5*obj_bbox[:, 2])
        dyc = abs(e_ped_bbox[:, 1]+0.5*e_ped_bbox[:, 3] - obj_bbox[:, 1]-0.5*obj_bbox[:, 3])
        w = obj_bbox[:, 2]
        h = obj_bbox[:, 3]
        # u: size: (n_objs, 8)
        pos_vec = torch.stack([dxmin, dymin, dxmax, dymax, dxc, dyc, w, h], -1)
        # size: (n_objs, 64)
        embed_pos = self.relu(self.pos_embed(pos_vec))
  
        ped_pos = torch.cat([curr_ped.repeat([n_objs, 1]), embed_pos], -1)
        embed_pedPos = self.relu(self.ped_embed(ped_pos))
        embed_objs = self.relu(self.ctxt_embed(curr_objs))
        ws = self.sigmoid((embed_objs * embed_pedPos).sum(1))
      A[1:, 0] = ws
      A[0, 1:] = ws
    else:
      raise NotImplementedError("Currently only support adj_type='uniform'.")
    return A


  def forward(self, ped, ctxt, ped_bbox=None, act=None, pose=None, obj_bbox=None, obj_cls=None, frames=None):
    B, T = len(ped), len(ped[0])

    # if ped.dim() == 5: # i.e. (B, T, 3, 224, 224)
    #   # extract features from backbone
    #   ped_feats, ctxt_feats = self.backbone_feats(ped, ctxt)
    # else:
    ped_feats, ctxt_feats = ped, ctxt

    # encode the frames as center nodes
    frames = frames.type(self.dtype).to(self.device)
    frames = frames.view(-1, 3, 224, 224)
    frame_enc = self.frame_encoder(frames)
    frame_enc = frame_enc.view(B, T, -1)


    # if ped_bbox is not None:
    #   # ped_bbox: (B, T, 4)
    #   ped_bbox = ped_bbox.to(self.device)
    if act is not None:
      act = act.to(self.device)

    # ped_feats, ctxt_feats: list of: (B, T, (n_objs, D))

    frame_feats = []
    for b in range(B):
      frame_feats += [],
      h_frame = None
      h_ctxt = None
      for t in range(T):
        curr_frame_enc = frame_enc[b][t]
        # frame_feats[b] += [],
        if len(ctxt_feats[b][t]) == 0 and len(ped_feats[b][t]) == 0:
          frame_feat = curr_frame_enc
        else:
          if len(ctxt_feats[b][t]) == 0:
            curr_nodes = ped_feats[b][t]
          elif len(ped_feats[b][t]) == 0:
            curr_nodes = ctxt_feats[b][t]
          else:
            curr_nodes = torch.cat([ped_feats[b][t], ctxt_feats[b][t]], 0)
          curr_nodes = curr_nodes.to(self.device)
          n_objs = curr_nodes.shape[0]
          # curr_ped = ped_feats[b][t].to(self.device)
          # curr_objs = ctxt_feats[b][t].to(self.device)
          # if curr_ped.dim() == 1:
          #   curr_ped = curr_ped.unsqueeze(0)
          # if curr_objs.dim() == 1:
          #   curr_objs = curr_objs.unsqueeze(0)
          # n_objs = curr_objs.shape[0]

          # curr_bbox = obj_bbox[b][t]

          # if n_objs != curr_bbox.shape[0]:
          #   if n_objs == 1 and curr_bbox.shape[0]==0:
          #     curr_bbox = np.zeros([1,4])
          #   else:
          #     pdb.set_trace()

          # if self.use_ctxt_gru:
          #   curr_ctxt = curr_objs.mean(0, keepdim=True)
          #   curr_objs = torch.cat([curr_objs, curr_ctxt], 0)
          #   n_objs += 1
          #   curr_bbox = np.concatenate([curr_bbox, np.zeros([1,4])], 0)

          # curr_bbox = torch.tensor(curr_bbox).type(self.dtype)

          for i in range(self.n_layers):
            # update adj mtrx
            A = self.getA(n_objs, frame_enc=curr_frame_enc, curr_nodes=curr_nodes).to(self.device)
            # graph conv
            X = torch.cat([curr_frame_enc.view(1, -1), curr_nodes], 0)
            X = self.W(torch.matmul(A, X))
            # update graph feats
            curr_frame_enc = X[:1]
            if self.use_frame_gru:
              curr_frame_enc = curr_frame_enc.unsqueeze(0) # size: (1, 1, 512)
              curr_frame_enc, h_frame = self.frame_gru(curr_frame_enc, h_frame)
              curr_frame_enc = curr_frame_enc.squeeze(0) # size: (1, 512)

            curr_nodes = X[1:]
            if self.use_node_gru:
              curr_nodes = curr_nodes[:-1]
              curr_ctxt = curr_nodes.mean(0, keepdim=True)
              curr_ctxt = curr_ctxt.unsqueeze(0) # size: (1, 1, 512)
              curr_ctxt, h_ctxt = self.node_gru(curr_ctxt, h_ctxt)
              curr_ctxt = curr_ctxt.squeeze(0) # size: (1, 512)
              curr_objs = torch.cat([curr_objs, curr_ctxt], 0)

          # if self.pos_mode != 'none':
          #   curr_ped = self.append_pos(curr_ped.unsqueeze(0), ped_bbox[b:b+1][t:t+1])
          #   curr_ped = curr_ped.squeeze(0)

          # if self.use_signal:
          #   # act 2: handwave / act 3: looking
          #   curr_ped = torch.cat([curr_ped, act[b][t:t+1][2:4]], -1)

          curr_ctxt = curr_nodes.mean(0, keepdim=True)
          frame_feat = torch.cat([curr_frame_enc, curr_ctxt], -1)
          # if self.branch == 'both':
          #   curr_ctxt  = curr_objs.mean(0, keepdim=True)
          #   frame_feat = torch.cat([curr_ped, curr_ctxt], -1)
          # else:
          #   frame_feat = curr_ped

        frame_feats[b] += frame_feat,
      # shape: (T, conv_dim)
      frame_feats[b] = torch.cat(frame_feats[b], 0).reshape(T, -1)
    # shape: (B, T, conv_dim)
    frame_feats = torch.stack(frame_feats, 0)
    frame_feats = frame_feats.to(self.device)

    # if self.use_gt_act:
    #   if self.n_acts == 1:
    #     act = act[:, :, 1:2]
    #   act = act[:, :self.seq_len]
    #   frame_feats = torch.cat([frame_feats, act], -1)

    if self.use_gru:
      # self.gru keeps the dimension of frame_feats
      frame_feats, h = self.gru(frame_feats)

    if self.predict:
      # o: (B, 1, cls_dim)
      o = frame_feats[:, -1:]
      pred_outs = []
      for pred_t in range(self.pred_seq_len):
        o, h = self.gru_pred(o, h)
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

    # shape: (B, T, 1)
    logits = self.classifier(frame_feats)

    return logits


