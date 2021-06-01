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

import copy
import math
import paddle
import paddle.nn as nn


def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


def constant_(tensor, val=0.):
    with paddle.no_grad():
        tensor.set_value(paddle.full_like(tensor, val))


def normal_(tensor, mean=0., std=1.):
    with paddle.no_grad():
        return tensor.set_value(paddle.normal(mean, std, tensor.shape))


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0, 0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def xavier_uniform_(tensor, gain=1.):
    # This is an implementation like pytorch
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    # Calculate uniform bounds from standard deviation
    bound = math.sqrt(3.0) * std
    return tensor.set_value(paddle.uniform(tensor.shape, min=-bound, max=bound))
