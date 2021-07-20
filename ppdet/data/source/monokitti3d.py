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
            
            if 'image' in self.data_fields:
                info['im_file'] = get_image_path(self.data_root, idx)
            
            if 'label' in self.data_fields:
                info.update({'label_path': get_label_path(self.data_root, idx)})
                info.update(get_label_anno(info['label_path']))
                
            if 'calib' in self.data_fields:
                info.update({'calib_path': get_calib_path(self.data_root, idx)})
                info.update(get_calib_info(info['calib_path']))
        
            if 'velodyne' in self.data_fields:
                info['velodyne_path'] = get_velodyne_path(self.data_root, idx)
            
            if 'label' in self.data_fields and 'calib' in self.data_fields:
                info.update(compute_centers(info))
            
            return info 
        
        
        with futures.ThreadPoolExecutor(self.num_worker) as executor:
            image_infos = executor.map(_parse_func, image_ids)
            
        self.roidbs = list(image_infos)
        

def compute_centers(info):
    '''center3d and center2d
    '''
    loc = info['location']
    dim = info['dimensions']
    # bbox = info['bbox']
    P0, P2 = info['P0'], info['P2']
    
    dst = np.array([0.5, 0.5, 0.5])
    src = np.array([0.5, 1.0, 0.5])
    center = loc + dim * (dst - src)
    
    offset = (P2[0, 3] - P0[0, 3]) / P2[0, 0]
    center3d = center.copy()
    center3d[0, 0] += offset # camera_2
    
    _center = np.concatenate((center, np.ones((center.shape[0], 1))), axis=-1)
    _center = (P2 @ _center.T).T
    _center = _center[:, [0, 1]] / _center[:, [2]]
    
    return {'center3d': center3d, 'center2d': _center}
    

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
    return _get_info_path(idx, os.path.join(prefix, 'velodyne'), '.bin')
    
    
def get_label_anno(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    annos = {}
    annos.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })

    content = [line.strip().split(' ') for line in lines]
    num_gt = len(content)
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    
    annos['name'] = np.array([x[0] for x in content])
    annos['truncated'] = np.array([float(x[1]) for x in content])
    annos['occluded'] = np.array([int(x[2]) for x in content])
    annos['alpha'] = np.array([float(x[3]) for x in content])
    annos['bbox'] = np.array([[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annos['dimensions'] = np.array([[float(info) for info in x[8:11]] for x in content]).reshape(-1, 3)[:, [2, 0, 1]]
    annos['location'] = np.array([[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annos['rotation_y'] = np.array([float(x[14]) for x in content]).reshape(-1)
    
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annos['score'] = np.array([float(x[15]) for x in content])
    else:
        annos['score'] = np.zeros((annos['bbox'].shape[0], ))
    
    # index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    # annos['index'] = np.array(index, dtype=np.int32)
    # annos['group_ids'] = np.arange(num_gt, dtype=np.int32)

    return annos


def _to_homo_coord(mat):
    _mat = np.eye(4)
    _j, _i = mat.shape
    _mat[:_j, :_i] = mat    
    return _mat

def get_calib_info(calib_path, use_homo_coord=True):
    '''P0 P1 P2 P3 R0_rect Tr_velo_to_cam Tr_imu_to_velo
    '''
    blobs = {}
    with open(calib_path, 'r') as f:
        lines = [lin.strip() for lin in f.readlines()]
        lines = [lin for lin in lines if len(lin) > 0]

        for lin in lines:
            items = [item.strip() for item in lin.strip().split(':')]
            values = np.array([float(x) for x in items[1].strip().split(' ')])

            matrix = values.reshape(3, 3) if items[0] == 'R0_rect' else values.reshape(3, 4)

            if use_homo_coord:
                blobs[items[0]] = _to_homo_coord(matrix)
            else:
                blobs[items[0]] = matrix
        
    return blobs

