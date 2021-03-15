import numpy as np
import os
import random
import torch

import args
from .logger import Logger


def build(is_train):
  opt, log = args.TrainArgs().parse() if is_train else args.TestArgs().parse()

  if not is_train:
    print('Options:')
    opt_dict = vars(opt)
    for key in sorted(opt_dict):
      print('{}: {}'.format(key, opt_dict[key]))
    if is_train:
      print('lr_init:', opt.lr_init)
      print('wd:', opt.wd)
      print('ckpt:', opt.ckpt_path)
    print()

  os.makedirs(opt.ckpt_path, exist_ok=True)

  # Set seed
  torch.manual_seed(2019)
  torch.cuda.manual_seed_all(2019)
  np.random.seed(2019)
  random.seed(2019)

  logger = Logger(opt.ckpt_path, opt.split)

  return opt, logger
