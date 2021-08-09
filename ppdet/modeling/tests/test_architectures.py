#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
<<<<<<< HEAD
import numpy as np

import paddle
import paddle.fluid as fluid
import os
import sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from ppdet.modeling.tests.decorator_helper import prog_scope
from ppdet.core.workspace import load_config, merge_config, create
=======
import ppdet
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9


class TestFasterRCNN(unittest.TestCase):
    def setUp(self):
        self.set_config()

    def set_config(self):
        self.cfg_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml'

    def test_trainer(self):
        # Trainer __init__ will build model and DataLoader
        # 'train' and 'eval' mode include dataset loading
        # use 'test' mode to simplify tests
        cfg = ppdet.core.workspace.load_config(self.cfg_file)
        trainer = ppdet.engine.Trainer(cfg, mode='test')


class TestMaskRCNN(TestFasterRCNN):
    def set_config(self):
        self.cfg_file = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml'


class TestCascadeRCNN(TestFasterRCNN):
    def set_config(self):
        self.cfg_file = 'configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.yml'


@unittest.skipIf(
    paddle.version.full_version < "1.8.4",
    "Paddle 2.0 should be used for YOLOv3 takes scale_x_y as inputs, "
    "disable this unittest for Paddle major version < 2")
class TestYolov3(TestFasterRCNN):
    def set_config(self):
        self.cfg_file = 'configs/yolov3/yolov3_darknet53_270e_coco.yml'


class TestSSD(TestFasterRCNN):
    def set_config(self):
        self.cfg_file = 'configs/ssd/ssd_vgg16_300_240e_voc.yml'


class TestGFL(TestFasterRCNN):
    def set_config(self):
        self.cfg_file = 'configs/gfl/gfl_r50_fpn_1x_coco.yml'


class TestPicoDet(TestFasterRCNN):
    def set_config(self):
        self.cfg_file = 'configs/picodet/picodet_s_shufflenetv2_320_coco.yml'


if __name__ == '__main__':
    unittest.main()
