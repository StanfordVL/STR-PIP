from .base_args import BaseArgs


class TestArgs(BaseArgs):
  def __init__(self):
    super(TestArgs, self).__init__()

    self.is_train = False
    # self.split = 'test'

    self.parser.add_argument('--mode', type=str, choices=['extract', 'evaluate'])

    # hyperparameters
    self.parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    self.parser.add_argument('--which-epoch', type=int,
                             help='which epochs to evaluate, -1 to load the best checkpoint')
    self.parser.add_argument('--slide', type=int, default=1,
                             help='Whether to use sliding window when testing.')
    self.parser.add_argument('--collect-A', type=int, default=0,
                             help="Whether to collect weight matrices in the graph model.")
    self.parser.add_argument('--save-As-format', type=str, default="",
                             help="Path to saved weight matrices.")
    # self.parser.add_argument('--rand-loader', type=int, default=0,
    #                          help="Whether to randomize the data loader.")

