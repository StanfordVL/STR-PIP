import torch.nn as nn
import math
import torch
import pdb
try:
  from models.backbone.resnet.basicblock import BasicBlock2D
  from models.backbone.resnet.bottleneck import Bottleneck2D
  # from utils.other import transform_input
  # from utils.meter import *
except:
  from resnet.basicblock import BasicBlock2D
  from resnet.bottleneck import Bottleneck2D
  # from basemodel.utils.other import transform_input
  # from basemodel.utils.meter import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

K_1st_CONV = 3


class ResNetBackBone(nn.Module):
    def __init__(self, blocks, layers,
                 str_first_conv='2D',
                 nb_temporal_conv=1,
                 list_stride=[1, 2, 2, 2],
                 **kwargs):
        self.nb_temporal_conv = nb_temporal_conv
        self.inplanes = 64
        super(ResNetBackBone, self).__init__()
        self._first_conv(str_first_conv)
        self.relu = nn.ReLU(inplace=True)
        self.list_channels = [64, 128, 256, 512]
        self.list_inplanes = []
        self.list_inplanes.append(self.inplanes)  # store the inplanes after layer1
        self.layer1 = self._make_layer(blocks[0], self.list_channels[0], layers[0], stride=list_stride[0])
        self.list_inplanes.append(self.inplanes)  # store the inplanes after layer1
        self.layer2 = self._make_layer(blocks[1], self.list_channels[1], layers[1], stride=list_stride[1])
        self.list_inplanes.append(self.inplanes)  # store the inplanes after layer2
        self.layer3 = self._make_layer(blocks[2], self.list_channels[2], layers[2], stride=list_stride[2])
        self.list_inplanes.append(self.inplanes)  # store the inplanes after layer3
        self.layer4 = self._make_layer(blocks[3], self.list_channels[3], layers[3], stride=list_stride[3])
        self.avgpool, self.avgpool_space, self.avgpool_time = None, None, None

        # Init of the weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _first_conv(self, str):
        self.conv1_1t = None
        self.bn1_1t = None
        if str == '3D_stabilize':
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(K_1st_CONV, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3),
                                   bias=False)
            self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(64)


        elif str == '2.5D_stabilize':
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                                   bias=False)
            self.conv1_1t = nn.Conv3d(64, 64, kernel_size=(K_1st_CONV, 1, 1), stride=(1, 1, 1),
                                      padding=(1, 0, 0),
                                      bias=False)
            self.bn1_1t = nn.BatchNorm3d(64)
            self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(64)

        elif str == '2D':
            self.conv1 = nn.Conv2d(3, 64,
                                   kernel_size=(7, 7),
                                   stride=(2, 2),
                                   padding=(3, 3),
                                   bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.bn1 = nn.BatchNorm2d(64)

        else:
            raise NameError

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None

        # Upgrade the stride is spatio-temporal kernel
        if not (block == BasicBlock2D or block == Bottleneck2D):
            stride = (1, stride, stride)

        if stride != 1 or self.inplanes != planes * block.expansion:
            if block is BasicBlock2D or block is Bottleneck2D:
                conv, batchnorm = nn.Conv2d, nn.BatchNorm2d
            else:
                conv, batchnorm = nn.Conv3d, nn.BatchNorm3d

            downsample = nn.Sequential(
                conv(self.inplanes, planes * block.expansion,
                     kernel_size=1, stride=stride, bias=False, dilation=dilation),
                batchnorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, dilation, nb_temporal_conv=self.nb_temporal_conv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nb_temporal_conv=self.nb_temporal_conv))

        return nn.Sequential(*layers)

    def forward(self, x, num=4) :
        # pdb.set_trace()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.conv1_1t is not None:
            x = self.conv1_1t(x)
            x = self.bn1_1t(x)
            x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        self.avgpool = nn.AvgPool2d((x.size(-1), x.size(-1))) if self.avgpool is None else self.avgpool
        x = self.avgpool(x)

        # Final classifier
        # x = x.view(x.size(0), -1)
        # x = self.fc_classifier(x)

        return x
