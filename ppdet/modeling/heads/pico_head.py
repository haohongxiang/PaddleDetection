# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from ppdet.core.workspace import register
from ppdet.modeling.layers import ConvNormLayer
from .simota_head import OTAVFLHead

from ..necks.csp_pan import ConvBNLayer
from ..initializer import normal_, constant_, bias_init_with_prob, uniform_

# class ConvBNLayer(nn.Layer):
#     def __init__(self,
#                  in_channel=96,
#                  out_channel=96,
#                  kernel_size=3,
#                  stride=1,
#                  groups=1,
#                  act='leaky_relu'):
#         super(ConvBNLayer, self).__init__()
#         initializer = nn.initializer.KaimingUniform()
#         self.act = act
#         assert self.act in ['leaky_relu', "hard_swish"]
#         self.conv = nn.Conv2D(
#             in_channels=in_channel,
#             out_channels=out_channel,
#             kernel_size=kernel_size,
#             groups=groups,
#             padding=(kernel_size - 1) // 2,
#             stride=stride,
#             weight_attr=ParamAttr(initializer=initializer),
#             bias_attr=False)
#         self.bn = nn.BatchNorm2D(out_channel)

#     def forward(self, x):
#         x = self.bn(self.conv(x))
#         if self.act == "leaky_relu":
#             x = F.leaky_relu(x)
#         elif self.act == "hard_swish":
#             x = F.hardswish(x)
#         return x


@register
class PicoFeatX(nn.Layer):
    """
    PicoFeat of PicoDet

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the LiteGFLFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
    """

    def __init__(self,
                 feat_in=[256, 512, 1024],
                 feat_out=256,
                 num_fpn_stride=3,
                 num_convs=2,
                 norm_type='bn',
                 share_cls_reg=False,
                 act='mish',
                 kernel_size=3,
                 negative_slope=0.01):
        super(PicoFeatX, self).__init__()

        self.num_convs = num_convs
        self.norm_type = norm_type
        self.share_cls_reg = share_cls_reg
        self.act = act
        self.cls_convs = []
        self.reg_convs = []
        for stage_idx in range(num_fpn_stride):
            cls_subnet_convs = []
            reg_subnet_convs = []
            for i in range(self.num_convs):
                in_c = feat_in[stage_idx] if i == 0 else feat_out
                cls_conv_dw = self.add_sublayer(
                    'cls_conv_dw{}.{}'.format(stage_idx, i),
                    ConvBNLayer(
                        in_c, feat_out, kernel_size, act=self.act))
                cls_subnet_convs.append(cls_conv_dw)

                if not self.share_cls_reg:
                    reg_conv_dw = self.add_sublayer(
                        'reg_conv_dw{}.{}'.format(stage_idx, i),
                        ConvBNLayer(
                            in_c, feat_out, kernel_size, act=self.act))
                    reg_subnet_convs.append(reg_conv_dw)

            self.cls_convs.append(cls_subnet_convs)
            self.reg_convs.append(reg_subnet_convs)

        self.negative_slope = negative_slope

    def act_func(self, x):
        if self.act == "leaky_relu":
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        elif self.act == 'mish':
            x = x * paddle.tanh(F.softplus(x))
        return x

    def forward(self, fpn_feat, stage_idx):
        assert stage_idx < len(self.cls_convs)
        cls_feat = fpn_feat
        reg_feat = fpn_feat

        if self.num_convs == 0:
            return cls_feat, reg_feat

        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = self.act_func(self.cls_convs[stage_idx][i](cls_feat))
            if not self.share_cls_reg:
                reg_feat = self.act_func(self.reg_convs[stage_idx][i](reg_feat))

        return cls_feat, reg_feat


@register
class PicoFeatL_cloud(nn.Layer):
    """
    PicoFeat of PicoDet

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the LiteGFLFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
    """

    def __init__(self,
                 feat_in=256,
                 feat_out=96,
                 num_fpn_stride=3,
                 num_convs=2,
                 norm_type='bn',
                 share_cls_reg=False,
                 act='hard_swish',
                 kernel_size=3,
                 negative_slope=0.01):

        super(PicoFeatL, self).__init__()
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.share_cls_reg = share_cls_reg
        self.act = act
        self.cls_convs = []
        self.reg_convs = []
        for stage_idx in range(num_fpn_stride):
            cls_subnet_convs = []
            reg_subnet_convs = []
            for i in range(self.num_convs):
                in_c = feat_in if i == 0 else feat_out
                # cls_conv_dw = self.add_sublayer(
                #     'cls_conv_dw{}.{}'.format(stage_idx, i),
                #     ConvBNLayer(
                #         in_c,
                #         feat_out,
                #         kernel_size,
                #         act=self.act,
                #         negative_slope=negative_slope))
                # cls_subnet_convs.append(cls_conv_dw)

                cls_conv_dw = self.add_sublayer(
                    'cls_conv_dw{}.{}'.format(stage_idx, i),
                    ConvNormLayer(
                        ch_in=in_c,
                        ch_out=feat_out,
                        filter_size=kernel_size,
                        stride=1,
                        # groups=feat_out,
                        norm_type=norm_type,
                        # bias_on=False,
                        lr_scale=1.))
                cls_subnet_convs.append(cls_conv_dw)

                # cls_conv_pw = self.add_sublayer(
                #     'cls_conv_pw{}.{}'.format(stage_idx, i),
                #     ConvNormLayer(
                #         ch_in=in_c,
                #         ch_out=feat_out,
                #         filter_size=1,
                #         stride=1,
                #         norm_type=norm_type,
                #         bias_on=False,
                #         lr_scale=2.))
                # cls_subnet_convs.append(cls_conv_pw)

                if not self.share_cls_reg:
                    # reg_conv_dw = self.add_sublayer(
                    #     'reg_conv_dw{}.{}'.format(stage_idx, i),
                    #     ConvBNLayer(
                    #         in_c,
                    #         feat_out,
                    #         kernel_size,
                    #         act=self.act,
                    #         negative_slope=negative_slope))
                    # reg_subnet_convs.append(reg_conv_dw)

                    reg_conv_dw = self.add_sublayer(
                        'reg_conv_dw{}.{}'.format(stage_idx, i),
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out,
                            filter_size=kernel_size,
                            stride=1,
                            # groups=feat_out,
                            norm_type=norm_type,
                            # bias_on=False,
                            lr_scale=1.))
                    reg_subnet_convs.append(reg_conv_dw)

                    # reg_conv_pw = self.add_sublayer(
                    #     'reg_conv_pw{}.{}'.format(stage_idx, i),
                    #     ConvNormLayer(
                    #         ch_in=in_c,
                    #         ch_out=feat_out,
                    #         filter_size=1,
                    #         stride=1,
                    #         norm_type=norm_type,
                    #         bias_on=False,
                    #         lr_scale=2.))
                    # reg_subnet_convs.append(reg_conv_pw)

            self.cls_convs.append(cls_subnet_convs)
            self.reg_convs.append(reg_subnet_convs)

        self.negative_slope = negative_slope

    def act_func(self, x):
        if self.act == "leaky_relu":
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        return x

    def forward(self, fpn_feat, stage_idx):
        assert stage_idx < len(self.cls_convs)
        cls_feat = fpn_feat
        reg_feat = fpn_feat
        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = self.act_func(self.cls_convs[stage_idx][i](cls_feat))
            if not self.share_cls_reg:
                reg_feat = self.act_func(self.reg_convs[stage_idx][i](reg_feat))
        return cls_feat, reg_feat


@register
class PicoFeatL(nn.Layer):
    """
    PicoFeat of PicoDet

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the LiteGFLFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
    """

    def __init__(self,
                 feat_in=256,
                 feat_out=96,
                 num_fpn_stride=3,
                 num_convs=2,
                 norm_type='bn',
                 share_cls_reg=False,
                 act='hard_swish',
                 kernel_size=3,
                 negative_slope=0.01,
                 lr_scale=1.0):

        super(PicoFeatL, self).__init__()

        assert act in ['leaky_relu', 'leaky', 'hard_swish', 'mish'], ''

        self.num_convs = num_convs
        self.norm_type = norm_type
        self.share_cls_reg = share_cls_reg
        self.act = act
        self.cls_convs = []
        self.reg_convs = []
        for stage_idx in range(num_fpn_stride):
            cls_subnet_convs = []
            reg_subnet_convs = []
            for i in range(self.num_convs):
                in_c = feat_in if i == 0 else feat_out
                # cls_conv_dw = self.add_sublayer(
                #     'cls_conv_dw{}.{}'.format(stage_idx, i),
                #     ConvBNLayer(
                #         in_c,
                #         feat_out,
                #         kernel_size,
                #         act=self.act,
                #         negative_slope=negative_slope))
                # cls_subnet_convs.append(cls_conv_dw)

                cls_conv_dw = self.add_sublayer(
                    'cls_conv_dw{}.{}'.format(stage_idx, i),
                    ConvNormLayer(
                        ch_in=in_c,
                        ch_out=feat_out,
                        filter_size=kernel_size,
                        stride=1,
                        # groups=feat_out,
                        norm_type=norm_type,
                        bias_on=False,
                        initializer=nn.initializer.KaimingUniform(),
                        lr_scale=lr_scale))
                cls_subnet_convs.append(cls_conv_dw)

                # cls_conv_pw = self.add_sublayer(
                #     'cls_conv_pw{}.{}'.format(stage_idx, i),
                #     ConvNormLayer(
                #         ch_in=in_c,
                #         ch_out=feat_out,
                #         filter_size=1,
                #         stride=1,
                #         norm_type=norm_type,
                #         bias_on=False,
                #         lr_scale=2.))
                # cls_subnet_convs.append(cls_conv_pw)

                if not self.share_cls_reg:
                    # reg_conv_dw = self.add_sublayer(
                    #     'reg_conv_dw{}.{}'.format(stage_idx, i),
                    #     ConvBNLayer(
                    #         in_c,
                    #         feat_out,
                    #         kernel_size,
                    #         act=self.act,
                    #         negative_slope=negative_slope))
                    # reg_subnet_convs.append(reg_conv_dw)

                    reg_conv_dw = self.add_sublayer(
                        'reg_conv_dw{}.{}'.format(stage_idx, i),
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out,
                            filter_size=kernel_size,
                            stride=1,
                            # groups=feat_out,
                            norm_type=norm_type,
                            bias_on=False,
                            initializer=nn.initializer.KaimingUniform(),
                            lr_scale=lr_scale))
                    reg_subnet_convs.append(reg_conv_dw)

                    # reg_conv_pw = self.add_sublayer(
                    #     'reg_conv_pw{}.{}'.format(stage_idx, i),
                    #     ConvNormLayer(
                    #         ch_in=in_c,
                    #         ch_out=feat_out,
                    #         filter_size=1,
                    #         stride=1,
                    #         norm_type=norm_type,
                    #         bias_on=False,
                    #         lr_scale=2.))
                    # reg_subnet_convs.append(reg_conv_pw)

            self.cls_convs.append(cls_subnet_convs)
            self.reg_convs.append(reg_subnet_convs)

        self.negative_slope = negative_slope

    def act_func(self, x):
        if self.act == "leaky_relu" or self.act == 'leaky':
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        elif self.act == 'mish':
            x = x * paddle.tanh(F.softplus(x))

        return x

    def forward(self, fpn_feat, stage_idx):
        assert stage_idx < len(self.cls_convs)
        cls_feat = fpn_feat
        reg_feat = fpn_feat

        if self.num_convs == 0:
            return cls_feat, reg_feat

        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = self.act_func(self.cls_convs[stage_idx][i](cls_feat))
            if not self.share_cls_reg:
                reg_feat = self.act_func(self.reg_convs[stage_idx][i](reg_feat))
        return cls_feat, reg_feat


@register
class PicoFeat(nn.Layer):
    """
    PicoFeat of PicoDet

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the LiteGFLFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
    """

    def __init__(self,
                 feat_in=256,
                 feat_out=96,
                 num_fpn_stride=3,
                 num_convs=2,
                 norm_type='bn',
                 share_cls_reg=False,
                 act='hard_swish'):
        super(PicoFeat, self).__init__()
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.share_cls_reg = share_cls_reg
        self.act = act
        self.cls_convs = []
        self.reg_convs = []
        for stage_idx in range(num_fpn_stride):
            cls_subnet_convs = []
            reg_subnet_convs = []
            for i in range(self.num_convs):
                in_c = feat_in if i == 0 else feat_out
                cls_conv_dw = self.add_sublayer(
                    'cls_conv_dw{}.{}'.format(stage_idx, i),
                    ConvNormLayer(
                        ch_in=in_c,
                        ch_out=feat_out,
                        filter_size=5,
                        stride=1,
                        groups=feat_out,
                        norm_type=norm_type,
                        bias_on=False,
                        lr_scale=2.))
                cls_subnet_convs.append(cls_conv_dw)
                cls_conv_pw = self.add_sublayer(
                    'cls_conv_pw{}.{}'.format(stage_idx, i),
                    ConvNormLayer(
                        ch_in=in_c,
                        ch_out=feat_out,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        bias_on=False,
                        lr_scale=2.))
                cls_subnet_convs.append(cls_conv_pw)

                if not self.share_cls_reg:
                    reg_conv_dw = self.add_sublayer(
                        'reg_conv_dw{}.{}'.format(stage_idx, i),
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out,
                            filter_size=5,
                            stride=1,
                            groups=feat_out,
                            norm_type=norm_type,
                            bias_on=False,
                            lr_scale=2.))
                    reg_subnet_convs.append(reg_conv_dw)
                    reg_conv_pw = self.add_sublayer(
                        'reg_conv_pw{}.{}'.format(stage_idx, i),
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out,
                            filter_size=1,
                            stride=1,
                            norm_type=norm_type,
                            bias_on=False,
                            lr_scale=2.))
                    reg_subnet_convs.append(reg_conv_pw)
            self.cls_convs.append(cls_subnet_convs)
            self.reg_convs.append(reg_subnet_convs)

    def act_func(self, x):
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        return x

    def forward(self, fpn_feat, stage_idx):
        assert stage_idx < len(self.cls_convs)
        cls_feat = fpn_feat
        reg_feat = fpn_feat
        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = self.act_func(self.cls_convs[stage_idx][i](cls_feat))
            if not self.share_cls_reg:
                reg_feat = self.act_func(self.reg_convs[stage_idx][i](reg_feat))
        return cls_feat, reg_feat


@register
class FeatHeadX(nn.Layer):
    def __init__(
            self,
            feat_in=[256, 512, 1024],
            feat_out=256,
            num_convs=2,
            num_stages=3,
            share_cls_reg=False,
            data_format='NCHW',
            act='mish', ):

        super(FeatHeadX, self).__init__()

        feat_in = feat_in if isinstance(feat_in, (
            list, tuple)) else [feat_in, ] * num_stages
        feat_out = feat_out if isinstance(feat_out, (
            list, tuple)) else [feat_out, ] * len(feat_in)

        self.in_channels = feat_in
        self.feat_channels = feat_out
        self.share_cls_reg = share_cls_reg

        self.stem_conv = nn.LayerList()
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()

        from ..backbones.darknet import ConvBNLayer
        for i, in_channel in enumerate(self.in_channels):
            self.stem_conv.append(
                ConvBNLayer(
                    ch_in=in_channel,
                    ch_out=self.feat_channels[i],
                    filter_size=1,
                    act=act,
                    data_format=data_format))

            self.cls_convs.append(
                nn.Sequential(*[
                    ConvBNLayer(
                        ch_in=self.feat_channels[i],
                        ch_out=self.feat_channels[i],
                        filter_size=3,
                        padding=1,
                        act=act,
                        data_format=data_format) for _ in range(num_convs)
                ]))

            if not share_cls_reg:
                self.reg_convs.append(
                    nn.Sequential(*[
                        ConvBNLayer(
                            ch_in=self.feat_channels[i],
                            ch_out=self.feat_channels[i],
                            filter_size=3,
                            padding=1,
                            act=act,
                            data_format=data_format) for _ in range(num_convs)
                    ]))

        # self._init_weights()

        # def _init_weights(self):
        #     bias_cls = bias_init_with_prob(0.01)
        #     for cls_head in zip(self.cls_convs):
        #         constant_(cls_head[-1].weight)
        #         constant_(cls_head[-1].bias, bias_cls)

    def forward(self, fpn_feat, idx):
        assert idx < len(self.cls_convs)

        fpn_feat = self.stem_conv[idx](fpn_feat)

        cls_feat = self.cls_convs[idx](fpn_feat)
        if not self.share_cls_reg:
            reg_feat = self.reg_convs[idx](fpn_feat)

            return cls_feat, reg_feat

        return cls_feat, cls_feat


@register
class FeatHeadL(nn.Layer):
    """
    PicoFeat of PicoDet

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the LiteGFLFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
    """

    def __init__(self,
                 feat_in=256,
                 feat_out=96,
                 num_fpn_stride=3,
                 num_convs=2,
                 norm_type='bn',
                 share_cls_reg=False,
                 act='mish',
                 kernel_size=3,
                 negative_slope=0.01,
                 lr_scale=1.0,
                 use_stem=False,
                 share_stem=True):

        super(FeatHeadL, self).__init__()
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.share_cls_reg = share_cls_reg
        self.act = act
        self.use_stem = use_stem
        self.share_stem = share_stem

        feat_in = feat_in if isinstance(feat_in, (
            list, tuple)) else [feat_in, ] * num_fpn_stride
        feat_out = feat_out if isinstance(feat_out, (
            list, tuple)) else [feat_out, ] * len(feat_in)

        self.cls_convs = []
        self.reg_convs = []
        self.stem_convs = []

        for stage_idx in range(num_fpn_stride):

            stem_subnet_convs = []
            if use_stem:
                if share_stem:
                    stem_conv = self.add_sublayer(
                        'stem_conv{}'.format(stage_idx),
                        ConvNormLayer(
                            ch_in=feat_in[stage_idx],
                            ch_out=feat_out[stage_idx],
                            filter_size=1,
                            stride=1,
                            norm_type=norm_type,
                            bias_on=False,
                            initializer=nn.initializer.KaimingUniform(),
                            lr_scale=lr_scale))
                    stem_subnet_convs.append(stem_conv)

                else:
                    stem_conv1 = self.add_sublayer(
                        'stem_conv1.{}'.format(stage_idx),
                        ConvNormLayer(
                            ch_in=feat_in[stage_idx],
                            ch_out=feat_out[stage_idx],
                            filter_size=1,
                            stride=1,
                            norm_type=norm_type,
                            bias_on=False,
                            initializer=nn.initializer.KaimingUniform(),
                            lr_scale=lr_scale))
                    stem_subnet_convs.append(stem_conv1)

                    stem_conv2 = self.add_sublayer(
                        'stem_conv2.{}'.format(stage_idx),
                        ConvNormLayer(
                            ch_in=feat_in[stage_idx],
                            ch_out=feat_out[stage_idx],
                            filter_size=1,
                            stride=1,
                            norm_type=norm_type,
                            bias_on=False,
                            initializer=nn.initializer.KaimingUniform(),
                            lr_scale=lr_scale))
                    stem_subnet_convs.append(stem_conv2)

            self.stem_convs.append(stem_subnet_convs)

            cls_subnet_convs = []
            reg_subnet_convs = []

            for i in range(self.num_convs):
                in_c = feat_in[stage_idx] if i == 0 else feat_out[stage_idx]

                if use_stem:
                    in_c = feat_out[stage_idx]

                # cls_conv_dw = self.add_sublayer(
                #     'cls_conv_dw{}.{}'.format(stage_idx, i),
                #     ConvBNLayer(
                #         in_c,
                #         feat_out,
                #         kernel_size,
                #         act=self.act,
                #         negative_slope=negative_slope))
                # cls_subnet_convs.append(cls_conv_dw)

                cls_conv_dw = self.add_sublayer(
                    'cls_conv_dw{}.{}'.format(stage_idx, i),
                    ConvNormLayer(
                        ch_in=in_c,
                        ch_out=feat_out[stage_idx],
                        filter_size=kernel_size,
                        stride=1,
                        # groups=feat_out,
                        norm_type=norm_type,
                        bias_on=False,
                        initializer=nn.initializer.KaimingUniform(),
                        lr_scale=lr_scale))
                cls_subnet_convs.append(cls_conv_dw)

                # cls_conv_pw = self.add_sublayer(
                #     'cls_conv_pw{}.{}'.format(stage_idx, i),
                #     ConvNormLayer(
                #         ch_in=in_c,
                #         ch_out=feat_out,
                #         filter_size=1,
                #         stride=1,
                #         norm_type=norm_type,
                #         bias_on=False,
                #         lr_scale=2.))
                # cls_subnet_convs.append(cls_conv_pw)

                if not self.share_cls_reg:
                    # reg_conv_dw = self.add_sublayer(
                    #     'reg_conv_dw{}.{}'.format(stage_idx, i),
                    #     ConvBNLayer(
                    #         in_c,
                    #         feat_out,
                    #         kernel_size,
                    #         act=self.act,
                    #         negative_slope=negative_slope))
                    # reg_subnet_convs.append(reg_conv_dw)

                    reg_conv_dw = self.add_sublayer(
                        'reg_conv_dw{}.{}'.format(stage_idx, i),
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out[stage_idx],
                            filter_size=kernel_size,
                            stride=1,
                            # groups=feat_out,
                            norm_type=norm_type,
                            bias_on=False,
                            initializer=nn.initializer.KaimingUniform(),
                            lr_scale=lr_scale))
                    reg_subnet_convs.append(reg_conv_dw)

                    # reg_conv_pw = self.add_sublayer(
                    #     'reg_conv_pw{}.{}'.format(stage_idx, i),
                    #     ConvNormLayer(
                    #         ch_in=in_c,
                    #         ch_out=feat_out,
                    #         filter_size=1,
                    #         stride=1,
                    #         norm_type=norm_type,
                    #         bias_on=False,
                    #         lr_scale=2.))
                    # reg_subnet_convs.append(reg_conv_pw)

            self.cls_convs.append(cls_subnet_convs)
            self.reg_convs.append(reg_subnet_convs)

        self.negative_slope = negative_slope

    def act_func(self, x):
        if self.act == "leaky_relu" or self.act == 'leaky':
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        elif self.act == 'mish':
            x = x * paddle.tanh(F.softplus(x))

        return x

    def forward(self, fpn_feat, stage_idx):
        assert stage_idx < len(self.cls_convs)

        if self.use_stem:
            if self.share_stem:
                assert len(self.stem_convs[stage_idx]) == 1, ''
                fpn_feat = self.act_func(self.stem_convs[stage_idx][0](
                    fpn_feat))
                cls_feat = fpn_feat
                reg_feat = fpn_feat
            else:
                assert len(self.stem_convs[stage_idx]) == 2, ''
                cls_feat = self.act_func(self.stem_convs[stage_idx][0](
                    fpn_feat))
                reg_feat = self.act_func(self.stem_convs[stage_idx][1](
                    fpn_feat))

        else:
            cls_feat = fpn_feat
            reg_feat = fpn_feat

        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = self.act_func(self.cls_convs[stage_idx][i](cls_feat))
            if not self.share_cls_reg:
                reg_feat = self.act_func(self.reg_convs[stage_idx][i](reg_feat))

        return cls_feat, reg_feat


@register
class PicoHead(OTAVFLHead):
    """
    PicoHead
    Args:
        conv_feat (object): Instance of 'PicoFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_class (object): Instance of VariFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        assigner (object): Instance of label assigner.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 7.
    """
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl', 'loss_bbox',
        'assigner', 'nms'
    ]
    __shared__ = ['num_classes']

    def __init__(self,
                 conv_feat='PicoFeat',
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32],
                 prior_prob=0.01,
                 loss_class='VariFocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 assigner='SimOTAAssigner',
                 reg_max=16,
                 feat_in_chan=96,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0,
                 kernel_size=1,
                 lr_scale=1):
        super(PicoHead, self).__init__(
            conv_feat=conv_feat,
            dgqp_module=dgqp_module,
            num_classes=num_classes,
            fpn_stride=fpn_stride,
            prior_prob=prior_prob,
            loss_class=loss_class,
            loss_dfl=loss_dfl,
            loss_bbox=loss_bbox,
            assigner=assigner,
            reg_max=reg_max,
            feat_in_chan=feat_in_chan,
            nms=nms,
            nms_pre=nms_pre,
            cell_offset=cell_offset)
        self.conv_feat = conv_feat
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_vfl = loss_class
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.assigner = assigner
        self.reg_max = reg_max
        self.feat_in_chan = [feat_in_chan, ] * len(fpn_stride) if isinstance(
            feat_in_chan, int) else feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset

        self.use_sigmoid = self.loss_vfl.use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        # Clear the super class initialization
        self.gfl_head_cls = None
        self.gfl_head_reg = None
        self.scales_regs = None

        self.head_cls_list = []
        self.head_reg_list = []
        for i in range(len(fpn_stride)):
            head_cls = self.add_sublayer(
                "head_cls" + str(i),
                nn.Conv2D(
                    in_channels=self.feat_in_chan[i],
                    out_channels=self.cls_out_channels + 4 * (self.reg_max + 1)
                    if self.conv_feat.share_cls_reg else self.cls_out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                    weight_attr=ParamAttr(
                        initializer=Normal(
                            mean=0., std=0.01),
                        learning_rate=lr_scale),
                    bias_attr=ParamAttr(
                        initializer=Constant(value=bias_init_value),
                        learning_rate=lr_scale)))
            self.head_cls_list.append(head_cls)
            if not self.conv_feat.share_cls_reg:
                head_reg = self.add_sublayer(
                    "head_reg" + str(i),
                    nn.Conv2D(
                        in_channels=self.feat_in_chan[i],
                        out_channels=4 * (self.reg_max + 1),
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        weight_attr=ParamAttr(
                            initializer=Normal(
                                mean=0., std=0.01),
                            learning_rate=lr_scale),
                        bias_attr=ParamAttr(
                            initializer=Constant(value=0),
                            learning_rate=lr_scale)))
                self.head_reg_list.append(head_reg)

    def forward(self, fpn_feats, deploy=False):
        assert len(fpn_feats) == len(
            self.fpn_stride
        ), "The size of fpn_feats is not equal to size of fpn_stride"
        cls_logits_list = []
        bboxes_reg_list = []
        for i, fpn_feat in enumerate(fpn_feats):
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat, i)
            if self.conv_feat.share_cls_reg:
                cls_logits = self.head_cls_list[i](conv_cls_feat)
                cls_score, bbox_pred = paddle.split(
                    cls_logits,
                    [self.cls_out_channels, 4 * (self.reg_max + 1)],
                    axis=1)
            else:
                cls_score = self.head_cls_list[i](conv_cls_feat)
                bbox_pred = self.head_reg_list[i](conv_reg_feat)

            if self.dgqp_module:
                quality_score = self.dgqp_module(bbox_pred)
                cls_score = F.sigmoid(cls_score) * quality_score

            if deploy:
                # Now only supports batch size = 1 in deploy
                # TODO(ygh): support batch size > 1
                cls_score = F.sigmoid(cls_score).reshape(
                    [1, self.cls_out_channels, -1]).transpose([0, 2, 1])
                bbox_pred = bbox_pred.reshape([1, (self.reg_max + 1) * 4,
                                               -1]).transpose([0, 2, 1])
            elif not self.training:
                cls_score = F.sigmoid(cls_score.transpose([0, 2, 3, 1]))
                bbox_pred = bbox_pred.transpose([0, 2, 3, 1])

            cls_logits_list.append(cls_score)
            bboxes_reg_list.append(bbox_pred)

        return (cls_logits_list, bboxes_reg_list)
