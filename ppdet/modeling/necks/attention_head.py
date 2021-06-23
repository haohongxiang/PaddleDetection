
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ppdet.core.workspace import serializable

from ..shape_spec import ShapeSpec
from .. import initializer as init


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
    def __init__(self, levels=3, channels=8, dim=128):
        super().__init__()
        
        L = levels
        C = channels
        dim = dim
        groups = 1
        k = 3
        
        self.k = k
        self.mid_idx = L // 2
        
        self.l_attention = nn.Sequential(nn.AdaptiveAvgPool2D(output_size=1),
                                        nn.Conv2D(L, L, 1, 1), 
                                        nn.ReLU(), 
                                        HardSigmoid(), )
        
        # self.offset_conv = nn.Conv2D(C, 2 * 3 * 3, 3, 1, 1)
        # self.weight_conv = nn.Sequential(nn.Conv2D(C, 3 * 3, 3, 1, 1), nn.Sigmoid())
        # self.deform_convs = nn.LayerList([paddle.vision.ops.DeformConv2D(C, C, 3, 1, 1) for i in range(L)])

        self.offset_conv = nn.Conv2D(C, k * k + 2 * k * k, 3, 1, 1)
        self.deform_conv = paddle.vision.ops.DeformConv2D(C * L, C * L, 3, 1, 1, groups=groups)

        self.c_attention = nn.Sequential(nn.AdaptiveAvgPool2D(output_size=1), 
                                         nn.Flatten(),  
                                         nn.Linear(C, dim), 
                                         nn.ReLU(), 
                                         nn.Linear(dim, C),
                                         nn.LayerNorm(C),
                                         ShiftedSigmoid(), )
        
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
        # offset = self.offset_conv(feat[:, self.mid_idx])
        # weight = self.weight_conv(feat[:, self.mid_idx])
        # sptials = []
        # for i in range(l):
        #     sptials.append(self.deform_convs[i](feat[:, i], offset, mask=weight))
        # feat = paddle.concat([s.unsqueeze(0) for s in sptials], axis=0).mean(axis=0)

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
    def __init__(self, in_channels, spatial_scales, num_heads=3, mid_layer_idx=None, hidden_dim=128):
        super().__init__()
                
        num_layers = len(in_channels)
        if mid_layer_idx is None:
            mid_layer_idx = num_layers // 2
        
        c = in_channels[mid_layer_idx]
        s = spatial_scales[mid_layer_idx]
        
        self.convs = nn.LayerList([nn.Conv2D(_c, c, 1, 1) for _c in in_channels])
        self.dyheads = nn.Sequential(*[DynamicHeadBlock(num_layers, c, hidden_dim) for _ in range(num_heads)])
        
        self.mid_layer_idx = mid_layer_idx
        self.num_layers = num_layers
        
        self.in_channels = in_channels
        self.out_channels = [c for _ in range(num_layers)]
        self.spatial_scales = [s for _ in range(num_layers)]
        
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
        feats = [self.convs[i](x) for i, x in enumerate(feats)]
        sizes = [x.shape[2:] for x in feats]
        feats = [F.interpolate(x, size=sizes[self.mid_layer_idx], mode='bilinear') for x in feats]
        
        feats = paddle.concat([x.unsqueeze(1) for x in feats], axis=1)
        feats = self.dyheads(feats)
        feats = [x.squeeze(1) for x in feats.split(self.num_layers, axis=1)]
        
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
        

