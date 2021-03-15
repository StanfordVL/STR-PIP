import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.backbone.resnet_based import resnet_backbone

from models.model_base import BaseModel

from itertools import accumulate
import pdb


class GraphModel(BaseModel):
  def __init__(self, opt):
    super().__init__(opt)
    if self.use_gru:
      self.gru = nn.GRU(self.cls_dim, self.cls_dim, 2, batch_first=True).to(self.device)
    if self.use_ped_gru:
      self.ped_gru = nn.GRU(self.conv_dim, self.conv_dim, 2, batch_first=True).to(self.device)
    if self.use_ctxt_gru:
      self.ctxt_gru = nn.GRU(self.conv_dim, self.conv_dim, 2, batch_first=True).to(self.device)
    if self.use_ctxt_node:
      self.ctxt_node_gru = nn.GRU(self.conv_dim, self.conv_dim, 2, batch_first=True).to(self.device)

    if self.use_pose:
      self.pose_embed = nn.Linear(self.ped_dim, self.conv_dim)

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

    self.use_obj_cls = opt.use_obj_cls

    # adjacency matrix
    self.adj_type = opt.adj_type
    if self.adj_type == 'inner' or self.adj_type == 'cosine' or self.adj_type == 'all':
      self.ped_embed = nn.Linear(self.conv_dim, 512)
      self.ctxt_embed = nn.Linear(self.conv_dim, 512)
      self.sm = nn.Softmax(-1)
    elif self.adj_type == 'spatial' or self.adj_type == 'spatialOnly':
      input_dim = 64
      if self.adj_type=='spatial':
        input_dim += self.conv_dim
      if self.use_obj_cls:
        input_dim += 1
      self.rel_embed = nn.Linear(input_dim, 128)
      # if self.use_obj_cls:
      #   self.ped_embed = nn.Linear(self.conv_dim+64+1, 128)
      # else:
      #   self.ped_embed = nn.Linear(self.conv_dim+64, 128)
      self.ctxt_embed = nn.Linear(self.conv_dim, 128)
      self.pos_embed = nn.Linear(8, 64)
    elif self.adj_type == 'prior':
      # TODO: save cls info (prior is defined over classes)
      raise NotImplementedError("'Prior' not implemented.")

    # layer weight
    self.diff_layer_weight = opt.diff_layer_weight
    if self.diff_layer_weight:
      layers = []
      for _ in range(self.n_layers):
        layers += nn.Linear(self.conv_dim, self.conv_dim),
      self.W = nn.ModuleList(layers)
    else:
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


  def getA(self, n_objs, curr_ped=None, curr_objs=None, ped_bbox=None, obj_bbox=None, obj_cls=None):
    def helper_get_pos_vec():
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
      return pos_vec


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
      embed_ped = self.ped_embed(curr_ped)
      embed_objs = self.ctxt_embed(curr_objs)
      if self.adj_type == 'inner':
        dot = torch.matmul(embed_ped, embed_objs.transpose(0,1))
        ws = self.sigmoid(dot)
        # normed_dot = (dot - dot.mean()) / (dot.max()-dot.min())
        # ws = self.sm(dot)
      elif self.adj_type == 'cosine':
        ws = F.cosine_similarity(embed_ped, embed_objs)
      A[1:, 0] = ws
      A[0, 1:] = ws
    elif self.adj_type == 'spatial' or self.adj_type == 'spatialOnly':
      # calculating rel positions
      # ped_bbox: (1, 4): (x_min, y_min, w, h)
      # obj_bbox: (n_objs, 4)
      obj_bbox = torch.tensor(obj_bbox).type(self.dtype).to(self.device)
      obj_cls = obj_cls.to(self.device)
      if obj_bbox.dim() == 1:
        # i.e. obj_bbox = tensor([])
        ws = torch.tensor([0])
      else:
        pos_vec = helper_get_pos_vec()
        # size: (n_objs, 64)
        embed_pos = self.relu(self.pos_embed(pos_vec))
  
        if self.adj_type == 'spatial':
          embed_pos = torch.cat([curr_ped.repeat([n_objs, 1]), embed_pos], -1)
        if self.use_obj_cls:
          embed_pos = torch.cat([embed_pos, obj_cls], -1)

        # if self.use_obj_cls:
        #   ped_pos = torch.cat([curr_ped.repeat([n_objs, 1]), embed_pos, obj_cls], -1)
        # else:
        #   ped_pos = torch.cat([curr_ped.repeat([n_objs, 1]), embed_pos], -1)

        try:
          # embed_pedPos = self.relu(self.ped_embed(ped_pos))
          embed_pos = self.relu(self.rel_embed(embed_pos))
        except Exception as e:
          print(e)
          pdb.set_trace()

        embed_objs = self.relu(self.ctxt_embed(curr_objs))
        # ws = self.sigmoid((embed_objs * embed_pedPos).sum(1))
        # ws = self.sigmoid((embed_objs * embed_pos).sum(1))
        ws = self.sigmoid((embed_objs * embed_pos).mean(1))
      A[1:, 0] = ws
      A[0, 1:] = ws
    elif self.adj_type == 'all':
      # (1, 512)
      embed_ped = self.ped_embed(curr_ped)
      # (n_objs, 512)
      embed_objs = self.ctxt_embed(curr_objs)
      nodes = torch.cat([embed_ped, embed_objs])
      # (1,1,.., 2,2,..., n,n,....)
      feat1 = nodes.unsqueeze(0).expand(n_objs+1, n_objs+1, 512)
      feat1 = feat1.transpose(1, 0).contiguous().view(-1, 512)
      # (1,2,..., n, 1,2,..., n, 1,2,...,n)
      feat2 = nodes.repeat([n_objs+1, 1])
      inner = (feat1 * feat2).sum(1)
      inner = self.sigmoid(inner)
      A = inner.view(n_objs+1, n_objs+1)
    else:
      raise NotImplementedError("Currently only support adj_type='uniform'.")
    return A


  def forward(self, ped, ctxt, ped_bbox=None, act=None, pose=None, obj_bbox=None, obj_cls=None, frames=None, collect_A=False):
    if ped.dim() == 5: # i.e. (B, T, 3, 224, 224)
      # extract features from backbone
      ped_feats, ctxt_feats = self.backbone_feats(ped, ctxt)
    else:
      ped_feats, ctxt_feats = ped.to(self.device), ctxt

    # ped_feats: (B, T, 1, D) if load_cache=='feats' else (B, T, D)
    # ctxt_feats: list of: (B, T, (n_objs, D))
    B, T = ped_feats.shape[:2]

    if self.use_pose:
      # pose: [B, T, 9, 3]
      pose = pose[:, :T, :, :-1]
      pose = pose.contiguous().view(B, T, 1, -1).to(self.device)
      ped_feats = torch.cat([ped_feats, pose], -1)
      ped_feats = self.pose_embed(ped_feats)
      # pdb.set_trace()

    if self.use_driver:
      frames = frames.type(self.dtype).to(self.device)
      frames = frames.view(-1, 3, 224, 224)
      # shape: [-1, 512, 1, 1]
      frame_enc = self.driver_encoder(frames)
      frame_enc = frame_enc.view(B, T, -1)
      # shape: [B, T, 512]
      frame_enc, _ = self.driver_gru(frame_enc)
      # shape: [B, T, 5]
      driver_act = self.driver_classifier(frame_enc)

    if ped_bbox is not None:
      # ped_bbox: (B, T, 4)
      ped_bbox = ped_bbox.to(self.device)
    if act is not None:
      act = act.to(self.device)

    frame_feats = []
    if collect_A:
      As = [[[] for _ in range(T)] for _ in range(B)]

    for b in range(B):
      frame_feats += [],
      h_ped = None
      h_ctxt = None
      h_ctxt_node = None
      for t in range(T):
        # frame_feats[b] += [],
        if len(ctxt_feats[b][t]) == 0:
          frame_feat = ped_feats[b][t]
        else:
          curr_ped = ped_feats[b][t].to(self.device)
          curr_objs = ctxt_feats[b][t].to(self.device)
          if curr_ped.dim() == 1:
            curr_ped = curr_ped.unsqueeze(0)
          if curr_objs.dim() == 1:
            curr_objs = curr_objs.unsqueeze(0)
          n_objs = curr_objs.shape[0]

          curr_bbox = obj_bbox[b][t]
          curr_obj_cls = obj_cls[b][t]

          if n_objs != curr_bbox.shape[0]:
            if n_objs == 1 and curr_bbox.shape[0]==0:
              curr_bbox = np.zeros([1,4])
              curr_obj_cls = np.zeros([1])
            else:
              pdb.set_trace()

          if self.use_ctxt_node:
            curr_ctxt = curr_objs.mean(0, keepdim=True)
            curr_objs = torch.cat([curr_objs, curr_ctxt], 0)
            n_objs += 1
            curr_bbox = np.concatenate([curr_bbox, np.zeros([1,4])], 0)
            curr_obj_cls = np.concatenate([curr_obj_cls, np.zeros([1])], 0)

          curr_bbox = torch.tensor(curr_bbox).type(self.dtype)
          curr_obj_cls = torch.tensor(curr_obj_cls).view(-1, 1).type(self.dtype)

          for i in range(self.n_layers):
            # update adj mtrx
            # pdb.set_trace()
            A = self.getA(n_objs, curr_ped, curr_objs, ped_bbox[b][t].unsqueeze(0), curr_bbox, curr_obj_cls).to(self.device)
            if collect_A:
              As[b][t] += A.cpu().detach().numpy(),
            # graph conv
            X = torch.cat([curr_ped, curr_objs], 0)
            if self.diff_layer_weight:
              X = self.W[i](torch.matmul(A, X))
            else:
              X = self.W(torch.matmul(A, X))
            # update graph feats
            curr_ped = X[:1]
            if self.use_ped_gru:
              curr_ped = curr_ped.unsqueeze(0) # size: (1, 1, 512)
              curr_ped, h_ped = self.ped_gru(curr_ped, h_ped)
              curr_ped = curr_ped.squeeze(0) # size: (1, 512)

            curr_objs = X[1:]
            if self.use_ctxt_node:
              curr_objs = curr_objs[:-1]
              curr_ctxt = curr_objs.mean(0, keepdim=True)
              curr_ctxt = curr_ctxt.unsqueeze(0) # size: (1, 1, 512)
              curr_ctxt, h_ctxt_node = self.ctxt_node_gru(curr_ctxt, h_ctxt_node)
              curr_ctxt = curr_ctxt.squeeze(0) # size: (1, 512)
              curr_objs = torch.cat([curr_objs, curr_ctxt], 0)

          if self.pos_mode != 'none':
            curr_ped = self.append_pos(curr_ped.unsqueeze(0), ped_bbox[b:b+1][t:t+1])
            curr_ped = curr_ped.squeeze(0)

          if self.use_signal:
            # act 2: handwave / act 3: looking
            curr_ped = torch.cat([curr_ped, act[b][t:t+1][2:4]], -1)

          if self.branch == 'both':
            curr_ctxt = curr_objs.mean(0, keepdim=True)
            if self.use_ctxt_gru:
              curr_ctxt = curr_ctxt.unsqueeze(0)
              curr_ctxt, h_ctxt = self.ctxt_gru(curr_ctxt, h_ctxt)
              curr_ctxt = curr_ctxt.squeeze(0)
            frame_feat = torch.cat([curr_ped, curr_ctxt], -1)
          else:
            frame_feat = curr_ped

        frame_feats[b] += frame_feat,
      # shape: (T, conv_dim)
      frame_feats[b] = torch.cat(frame_feats[b], 0).reshape(T, -1)
    # shape: (B, T, conv_dim)
    frame_feats = torch.stack(frame_feats, 0)
    frame_feats = frame_feats.to(self.device)

    if self.use_gt_act:
      if self.n_acts == 1:
        act = act[:, :, 1:2]
      act = act[:, :self.seq_len]
      frame_feats = torch.cat([frame_feats, act], -1)

    if self.use_driver:
      frame_feats = torch.cat([frame_feats, driver_act], -1)

    if self.use_gru:
      # self.gru keeps the dimension of frame_feats
      frame_feats, h = self.gru(frame_feats)
    else:
      h = None

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

    ret = {'logits': logits}
    if self.use_driver:
      ret['driver_act_logits'] = driver_act
    if collect_A:
      ret['As'] = As

    return ret


