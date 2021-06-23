
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ppdet.core.workspace import serializable

from ..shape_spec import ShapeSpec
from .. import initializer as init

from .yolo_fpn import YOLOv3FPN
from .fpn import FPN

class HardSigmoid(nn.Layer):
    def __init__(self, ):
        super().__init__()
        pass
    
    def forward(self, x):
        x = paddle.minimum(paddle.ones_like(x), (x + 1) / 2)
        x = paddle.maximum(paddle.zeros_like(x), x)
        return x
    
    
class ShiftedSigmoid(nn.Layer):
    '''shift [-1, 1]
    '''
    def __init__(self, ):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(x)
        return 2 * x - 1
        

class DynamicReLUB(nn.Layer):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super().__init__()
        
        self.channels = channels
        self.k = k
        self.conv_type = conv_type.lower()
        
        self.mm_coefs = nn.Sequential(nn.Linear(channels, channels // reduction),
                                      nn.ReLU(), 
                                      nn.Linear(channels // reduction, 2 * channels * k),
                                      ShiftedSigmoid())
        
        self.register_buffer('init_w', paddle.to_tensor([1.] * k + [0.5] * k, dtype='float32'))
        self.register_buffer('init_b', paddle.to_tensor([1.] + [0.] * (2 * k - 1), dtype='float32'))

    def get_params(self, x):
        # n, c, h, w -> n, c -> n, c * 2k
        theta = paddle.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = paddle.mean(theta, axis=-1)        
        return self.mm_coefs(theta)

    def forward(self, x):
        
        theta = self.get_params(x)
        theta = theta.reshape([-1, self.channels, 2 * self.k]) * self.init_w + self.init_b
    
        if self.conv_type == '2d':
            # n c h w -> h w n c 1
            out = x.transpose([2, 3, 0, 1]).unsqueeze(-1)
            out = out * theta[:, :, :self.k] + theta[:, :, self.k:]
            
            # n w n c 2 -> n c h w
            out = out.max(axis=-1).transpose([2, 3, 0, 1])
            
        return out


class DynamicHeadBlock(nn.Layer):
    def __init__(self, levels=3, channels=8, ):
        super().__init__()
        
        L = levels
        C = channels
        dim = 512
        groups = 1
        k = 3
        
        self.k = k
        self.mid_idx = L // 2
        
        self.l_attention = nn.Sequential(nn.AdaptiveAvgPool2D(output_size=1),
                                        nn.Conv2D(L, L, 1, 1), 
                                        nn.ReLU(), 
                                        HardSigmoid(), )
        
        self.offset_conv = nn.Conv2D(C, k * k + 2 * k * k, 3, 1, 1)
        self.deform_conv = paddle.vision.ops.DeformConv2D(C * L, C * L, k, 1, 1, groups=groups)

        self.dynamic_relu = nn.Sequential(nn.AdaptiveAvgPool2D(output_size=1), DynamicReLUB(C))
    
        init.reset_initialized_parameter(self)
        
    def forward(self, feat):
        '''
        feat [N, L, C, H, W]
        '''
        n, l, c, h, w = feat.shape
        
        # layer
        feat = feat.reshape([n, l, c, -1])
        feat = self.l_attention(feat) * feat
        feat = feat.reshape([n, l, c, h, w])
        # print('Layers Attention: ', feat.shape)
        
        # spatial
        output = self.offset_conv(feat[:, self.mid_idx])
        weight = F.sigmoid(output[:, :self.k * self.k])
        offset = output[:, self.k * self.k:]

        feat = feat.reshape([n, l * c, h, w])
        feat = self.deform_conv(feat, offset, mask=weight)
        feat = feat.reshape([n, l, c, h, w])
        # print('Spatial Attention: ', feat.shape)

        # channel 
        feat = feat.reshape([n, l, c, h * w]).transpose([0, 2, 1, 3])        
        feat = self.dynamic_relu(feat) * feat
        # print('Channel Attention: ', feat.shape)
        
        feat = feat.reshape([n, l, c, h, w])
        # print('Ouput: ', feat.shape)
        
        return feat


@register
class DynamicHead(nn.Layer):
    __shared__ = ['norm_type']
    
    def __init__(self, in_channels, spatial_scales, out_channels=512, num_heads=3, mid_layer_idx=None, norm_type='bn', use_fpn=False):
        super().__init__()
        
        num_layers = len(in_channels)
        if mid_layer_idx is None:
            mid_layer_idx = num_layers // 2
        
        c = out_channels # in_channels[mid_layer_idx]
        s = spatial_scales[mid_layer_idx]
        
        self.use_fpn = use_fpn
        if self.use_fpn:
            # self.fpn = YOLOv3FPN(in_channels=in_channels, norm_type=norm_type)
            self.fpn = FPN(in_channels=in_channels, out_channel=out_channels, extra_stage=0, norm_type=norm_type)
        else:
            self.in_convs = nn.LayerList([nn.Conv2D(_c, c, 1, 1) for _c in in_channels])

        self.dyheads = nn.Sequential(*[DynamicHeadBlock(num_layers, c) for _ in range(num_heads)])
        
        self.mid_layer_idx = mid_layer_idx
        self.num_layers = num_layers
        
        self.in_channels = in_channels
        self.out_channels = [out_channels for _ in range(num_layers)]
        self.spatial_scales = spatial_scales # [s for _ in range(num_layers)]
        
        init.reset_initialized_parameter(self)
        
    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'spatial_scales': [1.0 / i.stride for i in input_shape],
        }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c, stride=1./s) for c, s in zip(self.out_channels, self.spatial_scales)]

    def forward(self, feats, for_mot=False):
        
        if self.use_fpn:
            feats = self.fpn(feats)
        else:
            feats = [self.in_convs[i](x) for i, x in enumerate(feats)]
                    
        sizes = [x.shape[2:] for x in feats]
        feats = [F.interpolate(x, size=sizes[self.mid_layer_idx], mode='bilinear') for x in feats]
        feats = paddle.concat([x.unsqueeze(1) for x in feats], axis=1)
        
        feats = self.dyheads(feats)
        
        feats = [x.squeeze(1) for x in feats.split(self.num_layers, axis=1)]
        # feats = [self.out_convs[i](x) for i, x in enumerate(feats)]
        feats = [F.interpolate(x, size=sz, mode='bilinear') for x, sz in zip(feats, sizes)]
        
        return feats[::-1]

    
if __name__ == '__main__':
    
    m = DynamicHeadBlock()
    data = paddle.rand([2, 3, 8, 3, 3])
    m(data).sum().backward()
    
    
    channels = [8, 8, 16]
    sizes = [10, 16, 20]
    head = DynamicHead(channels, )
    data = [paddle.rand([2, c, s, s]) for c, s in zip(channels, sizes)]
    output = head(data)
    for out in output:
        print(out.shape)
        

