# -*- coding: utf-8 -*-



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


_BN_MOMENTUM = 0.1


def _conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(BasicBlock, self).__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):


    def __init__(self, block, layer_num_list, heads, head_conv_channels):

        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2,
            padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layer_num_list[0])
        self.layer2 = self._make_layer(block, 128, layer_num_list[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_num_list[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_num_list[3], stride=2)


        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
        )

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv_channels > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv_channels,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv_channels, num_output,
                              kernel_size=1, stride=1, padding=0))
            else:
                fc = nn.Conv2d(
                    in_channels=256,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(
                    planes * block.expansion, momentum=_BN_MOMENTUM
                ),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):

        deconv_cfg_dict = {
            2: (0, 0),
            3: (1, 1),
            4: (1, 0),
        }
        padding, output_padding = deconv_cfg_dict[deconv_kernel]
        return padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):

        if num_layers != len(num_filters):
            raise ValueError(
                "num of deconv layers {} ".format(num_layers),
                "and filters {} not equal".format(len(num_filters)))
        if num_layers != len(num_kernels):
            raise ValueError(
                "num of deconv layers {} ".format(num_layers),
                "and kernels {} not equal".format(len(num_kernels)))

        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


def get_resnet(layer_num, heads, head_conv_channels):

    resnet_params_dict = {
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
    }
    block_class, layer_num_list = resnet_params_dict[layer_num]
    model = ResNet(
        block_class, layer_num_list, heads,
        head_conv_channels=head_conv_channels
    )
    return model
