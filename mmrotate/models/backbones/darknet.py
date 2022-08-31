# Copyright (c) OpenMMLab. All rights reserved.
# Modified from csuhan: https://github.com/csuhan/ReDet
from mmcv.runner import BaseModule
from ..builder import ROTATED_BACKBONES
from ..utils.common import *


@ROTATED_BACKBONES.register_module()
class Darknet_YOLO(BaseModule):

    arch_settings = {
        'n': [0.33, 0.25],
        's': [0.33, 0.5],
        'm': [0.67, 0.75],
        'l': [1.0, 1.0],
        'x': [1.33, 1.25]
    }
    arch_body = (1, 1, 3, 1, 6, 1, 9, 1, 3, 1)
    arch_type = ('Conv', 'Conv', 'C3', 'Conv', 'C3', 'Conv', 'C3', 'Conv', 'C3', 'SPPF')
    arch_param = (
        [64, 6, 2, 2],
        [128, 3, 2],
        [128],
        [256, 3, 2],
        [256],
        [512, 3, 2],    #out1
        [512],
        [1024, 3, 2],   #out2
        [1024],
        [1024, 5]       #out3
    )

    def __init__(self,
                 arch ='s',
                 out_indices=(4, 6, 9),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 norm_eval=True,
                 pretrained=None,
                 init_cfg=None):
        super(Darknet_YOLO, self).__init__()

        self.arch = arch
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.depth, self.width = self.arch_settings[arch]

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        layers, save, ch = [], [], [3]
        for i, (n, m, args) in enumerate(zip(self.arch_body, self.arch_type, self.arch_param)):
            m = eval(m) if isinstance(m, str) else m

            n = n_ = max(round(n * self.depth), 1) if n > 1 else n  # depth gain
            if m in [Conv, SPPF, Bottleneck, BottleneckCSP, C3]:
                c1, c2 = ch[-1], args[0]
                c2 = make_divisible(c2 * self.width, 8)
                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3]:
                    args.insert(2, n)  # number of repeats
                    n = 1

                m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace('__main__.', '')
            np = sum(x.numel() for x in m_.parameters())  # number params
            m_.i, m_.f, m_.type, m_.np = i, -1, t, np  # attach index, 'from' index, type, number params
            layers.append(m_)
            ch.append(c2)
        self.model = nn.Sequential(*layers)

        self.norm_eval = norm_eval

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        for m in self.model:
            x = m(x)
            print( m.type + ' : ' + str(m.i))
            if m.i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

