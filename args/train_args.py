from .base_args import BaseArgs


class TrainArgs(BaseArgs):
  """
  Defines base arguments for Model Training
  """
  def __init__(self):
    super(TrainArgs, self).__init__()

    self.is_train = True
    # self.split = 'train'

    self.parser.add_argument('--batch-size', type=int, default=4, help='batch size per gpu')
    self.parser.add_argument('--n-epochs', type=int, default=50, help='total # of epochs')
    self.parser.add_argument('--n-iters', type=int, default=0, help='total # of iterations')
    self.parser.add_argument('--start-epoch', type=int, default=0, help='starting epoch')
    self.parser.add_argument('--lr-init', type=float, default=1e-3, help='initial learning rate')
    self.parser.add_argument('--lr-decay', type=int, default=0, choices=[0, 1], help='whether to decay learning rate')
    self.parser.add_argument('--decay-every', type=int, default=10)
    self.parser.add_argument('--wd', type=float, default=1e-5)
    self.parser.add_argument('--load-ckpt-dir', type=str, default='', help='directory of checkpoint')
    self.parser.add_argument('--load-ckpt-epoch', type=int, default=0, help='epoch to load checkpoint')
