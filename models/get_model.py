# ped-centric
from .model_concat import ConcatModel
from .model_graph import GraphModel
from .model_pos import PosModel
# loc-centric
from .model_loc_graph import LocGraphModel
from .model_loc_concat import LocConcatModel
# baselines
from .baseline_anticipate_cnn import BaselineAnticipateCNN
from .baseline_pose import BaselinePose


def get_model(opt):
  # ped-centric
  if opt.model == 'concat':
    model = ConcatModel(opt)
  elif opt.model == 'graph':
    model = GraphModel(opt)
  elif opt.model == 'pos':
    model = PosModel(opt)
  # loc-centric
  elif opt.model == 'loc_concat':
    model = LocConcatModel(opt)
  elif opt.model == 'loc_graph':
    model = LocGraphModel(opt)
  # baselines
  elif opt.model == 'baseline_anticipate_cnn':
    model = BaselineAnticipateCNN(opt)
  elif opt.model == 'baseline_pose':
    model = BaselinePose(opt)
  else:
    raise NotImplementedError

  model.setup()
  return model
