# Copyright (c) OpenMMLab. All rights reserved.
# Modified from csuhan: https://github.com/csuhan/ReDet
import warnings
from ..builder import ROTATED_NECKS
import math
from mmcv.runner import BaseModule
import torch.nn as nn
import torch
from ..utils.common import *

@ROTATED_NECKS.register_module()
class PAFPN_YOLO(BaseModule):
    arch_settings = {
        'n': [0.33, 0.25],
        's': [0.33, 0.5],
        'm': [0.67, 0.75],
        'l': [1.0, 1.0],
        'x': [1.33, 1.25]
    }

    def __init__(self,
                 arch='s',
                 in_channels=[256, 512, 1024]
                 ):
        super(PAFPN_YOLO, self).__init__()
        self.depth, self.width = self.arch_settings[arch]
        n = n_ = max(round(3 * self.depth), 1)  # depth gain
        self.in_channels = [make_divisible(c * self. width, 8) for c in in_channels]

        #x_channel: [256, 512, 1024]
        self.conv1 = Conv(self.in_channels[2], self.in_channels[1], 1, 1)
        self.upsample1 = nn.Upsample(None, 2,'nearest')
        self.concat1 = Concat()
        self.c3_1 = nn.Sequential(*[C3(c1=self.in_channels[2], c2=self.in_channels[1],shortcut=False) for _ in range(n)])

        self.conv2 = Conv(self.in_channels[1], self.in_channels[0], 1, 1)
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.concat2 = Concat()
        self.c3_2 = nn.Sequential(*[C3(c1=self.in_channels[1], c2=self.in_channels[0], shortcut=False) for _ in range(n)])

        self.conv3 = Conv(self.in_channels[0], self.in_channels[0], 3, 2)
        self.concat3 = Concat()
        self.c3_3 = nn.Sequential(*[C3(c1=self.in_channels[1], c2=self.in_channels[1], shortcut=False) for _ in range(n)])

        self.conv4 = Conv(self.in_channels[1], self.in_channels[1], 3, 2)
        self.concat4 = Concat()
        self.c3_4 = nn.Sequential(*[C3(c1=self.in_channels[2], c2=self.in_channels[2],shortcut=False) for _ in range(n)])


    def forward(self, x):
        x3, x2, x1 = x
        x1_conv1 =  self.conv1(x1)
        x1_upsample = self.upsample1(x1_conv1)
        xf1 = self.concat1([x2, x1_upsample])
        xf1_conv = self.conv2(self.c3_1(xf1))
        xf1_upsample = self.upsample2(xf1_conv)
        x1_out = self.c3_2(self.concat2([x3, xf1_upsample]))

        x1_out_conv = self.conv3(x1_out)
        xf2 = self.concat3([x1_out_conv, xf1_conv])
        x2_out = self.c3_3(xf2)

        x2_out_conv = self.conv4(x2_out)
        xf3 = self.concat4([x2_out_conv, x1_conv1])
        x3_out = self.c3_4(xf3)
        return tuple([x1_out, x2_out, x3_out])

