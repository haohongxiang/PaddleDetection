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

import os
import copy
import numpy as np
import concurrent.futures as futures

from ppdet.core.workspace import register, serializable
from .dataset import DetDataset

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['MonoKitti3d']


@register
@serializable
class MonoKitti3d(DetDataset):
    def __init__(self, dataset_dir, image_dir, anno_path, data_fields=['image'],
                 sample_num=1, use_default_label=None, num_worker=8, **kwargs):
        super().__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            use_default_label=use_default_label,
            **kwargs)
        
        [setattr(self, k, v) for k, v in locals().items()]
        
        self.anno_path = os.path.join(dataset_dir, anno_path)
        self.data_root = os.path.join(dataset_dir, image_dir)
        
        self.parse_dataset()
        
        
    def __getitem__(self, idx):
        # data batch
        roidb = copy.deepcopy(self.roidbs[idx])
        
        return roidb
    
    def parse_dataset(self):
        '''parse_dataset
        '''
        image_ids = [lin.strip() for lin in open(self.anno_path, 'r').readlines() if lin]
        print(self.data_root)
        
        def _parse_func(idx):
            info = {}
            info.update({'image_path': get_image_path(self.data_root, idx)})
            info.update({'label_path': get_label_path(self.data_root, idx)})
            info.update({'calib_path': get_calib_path(self.data_root, idx)})
        
            return info 
        
        
        with futures.ThreadPoolExecutor(self.num_worker) as executor:
            image_infos = executor.map(_parse_func, image_ids)
            
        self.roidbs = list(image_infos)
        
        

def _get_image_index_str(idx):
    return "{:0>6}".format(idx)
    
def _get_info_path(idx, prefix, suffix, check_exists=True):
    path = os.path.join(prefix, _get_image_index_str(idx) + suffix)
    if check_exists:
        assert os.path.exists(path), f'{path} does not exists.'
    return path
    
def get_image_path(prefix, idx):
    return _get_info_path(idx, os.path.join(prefix, 'image_2'), '.png')

def get_label_path(prefix, idx):
    return _get_info_path(idx, os.path.join(prefix, 'label_2'), '.txt')

def get_calib_path(prefix, idx):
    return _get_info_path(idx, os.path.join(prefix, 'calib'), '.txt')

def get_velodyne_path(prefix, idx):
    return _get_info_path(idx, os.path.join(prefix, 'calib'), '.txt')
