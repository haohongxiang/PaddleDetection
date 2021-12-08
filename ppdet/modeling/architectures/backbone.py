# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['Backbone']


@register
class Backbone(BaseArch):
    __category__ = 'architecture'

    def __init__(
            self,
            backbone, ):

        super(Backbone, self).__init__()
        self.backbone = backbone

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}

        return {'backbone': backbone, }

    def _forward(self):
        body_feats = self.backbone(self.inputs)

        if self.training:
            return sum([out.sum() for out in body_feats])
        else:
            return sum([out.sum() for out in body_feats]), sum(
                [out.sum() for out in body_feats])

    def get_loss(self, ):
        bbox_loss = self._forward()
        loss = {}
        loss.update(bbox_loss)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output
