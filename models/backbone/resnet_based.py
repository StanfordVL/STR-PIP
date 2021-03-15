import torch
try:
  from models.backbone.imagenet_pretraining import load_pretrained_2D_weights
  from models.backbone.resnet.basicblock import BasicBlock2D, BasicBlock3D, BasicBlock2_1D
  from models.backbone.resnet.bottleneck import Bottleneck2D, Bottleneck3D, Bottleneck2_1D
  from models.backbone.resnet.resnet import ResNetBackBone
except:
  from imagenet_pretraining import load_pretrained_2D_weights
  from resnet.basicblock import BasicBlock2D, BasicBlock3D, BasicBlock2_1D
  from resnet.bottleneck import Bottleneck2D, Bottleneck3D, Bottleneck2_1D
  from resnet.resnet import ResNetBackBone


# __all__ = [
#     'resnet_two_heads',
# ]


def resnet_backbone(depth=18, blocks='2D_2D_2D_2D', **kwargs):
    """Constructs a ResNet-18 model backbone
    """
    # Blocks and layers
    list_block, list_layers = get_cnn_structure(depth=depth,
                                                str_blocks=blocks)

    # Model with two heads
    model = ResNetBackBone(list_block,
                   list_layers,
                   **kwargs)

    if False:
      print(
          "*** Backbone: Resnet{} (blocks: {} - pooling: {} - Two heads - blocks 2nd head: {} and fm size 2nd head: {}) ***".format(
              depth,
              blocks,
              pooling,
              object_head,
              model.size_fm_2nd_head))

    # Pretrained from imagenet weights
    model = load_pretrained_2D_weights('resnet{}'.format(depth), model, inflation='center')

    return model


def get_cnn_structure(str_blocks='2D_2D_2D_2D', depth=18):
    # List of blocks
    list_block = []

    # layers
    if depth == 18:
        list_layers = [2, 2, 2, 2]
        nature_of_block = 'basic'
    elif depth == 34:
        list_layers = [3, 4, 6, 3]
        nature_of_block = 'basic'
    elif depth == 50:
        list_layers = [3, 4, 6, 3]
        nature_of_block = 'bottleneck'
    else:
        raise NameError

    # blocks
    if nature_of_block == 'basic':
        block_2D, block_3D, block_2_1D = BasicBlock2D, BasicBlock3D, BasicBlock2_1D
    elif nature_of_block == 'bottleneck':
        block_2D, block_3D, block_2_1D = Bottleneck2D, Bottleneck3D, Bottleneck2_1D
    else:
        raise NameError

    # From string to blocks
    list_block_id = str_blocks.split('_')

    # Catch from the options if exists
    for i, str_block in enumerate(list_block_id):
        # Block kind
        if str_block == '2D':
            list_block.append(block_2D)
        elif str_block == '2.5D':
            list_block.append(block_2_1D)
        elif str_block == '3D':
            list_block.append(block_3D)
        else:
            # ipdb.set_trace()
            raise NameError

    return list_block, list_layers


if __name__ == '__main__':
  model = resnet_backbone()
  img = torch.ones([1, 1, 224, 224])
  feats = model(img)
  print(feats.shape)
