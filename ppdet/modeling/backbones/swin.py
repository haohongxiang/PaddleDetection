import paddle
import paddle.nn as nn

from ppdet.core.workspace import register



@register
class WindowAttention(nn.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        print('qkv_bias', qkv_bias, 'qk_scale', qk_scale)
        
        