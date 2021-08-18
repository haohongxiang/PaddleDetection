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
import numpy as np
from ppdet.core.workspace import register, serializable
from .dataset import DetDataset

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


import concurrent.futures as futures


__all__ = ['PlainDetDataSet']



@register
@serializable
class PlainDetDataSet(DetDataset):
    """
    Load dataset with COCO format.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): coco annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth. 
            False as default
        allow_empty (bool): whether to load empty entry. False as default
        empty_ratio (float): the ratio of empty record number to total 
            record's, if empty_ratio is out of [0. ,1.), do not sample the 
            records. 1. as default
    
    dataset_dir
        anno_path_train_1.txt
        anno_path_train_2.txt
        anno_path_test_1.txt
        anno_path_test_2.txt
    
    im_file.jpg, class_id x1 y1 x2 y2, class_id x1 y1 x2 y2
    
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.):
        super(PlainDetDataSet, self).__init__(dataset_dir, image_dir, anno_path,
                                          data_fields, sample_num)
        self.load_image_only = False
        self.load_semantic = False
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio
        
        self.num_worker = 8
        self.dataset_dir = dataset_dir
        self.anno_path = anno_path
        
        self.parse_dataset()
        
        
    def _sample_empty(self, records, num):
        # if empty_ratio is out of [0. ,1.), do not sample the records
        if self.empty_ratio < 0. or self.empty_ratio >= 1.:
            return records
        import random
        sample_num = int(num * self.empty_ratio / (1 - self.empty_ratio))
        records = random.sample(records, sample_num)
        return records


    def parse_dataset(self):
        '''parse pain txt
        '''
        if not isinstance(self.anno_path, (list, tuple)):
            anno_paths = (self.anno_path, )
        else:
            anno_paths = self.anno_path
            
        print(self.dataset_dir)
        
        anno_paths = [os.path.join(self.dataset_dir, anno) for anno in anno_paths]

        lines = []
        for anno in anno_paths:
            lines.extend(open(anno, 'r').readlines())
            
        lines = [lin for lin in lines if lin]
                
        with futures.ThreadPoolExecutor(self.num_worker) as executor:
            roidbs = executor.map(self._parse_line, lines)
        
        self.roidbs = [t for t in roidbs if t]
        
        logger.warning('loading data done...')
        

    def _parse_line(self, lin):
        '''
        im_file.jpg, class_id x1 y1 x2 y2, class_id x1 y1 x2 y2
        '''
        items = lin.strip(' ,\t').split(',')
        
        if not os.path.exists(items[0]):
            logger.warning(f'{items[0]} not exist...')
            return None

        if len(items[1:]) == 0:
            logger.warning('empty..')
            return None

        annos = [_bbox.strip().split(' ') for _bbox in items[1:]]
        for _bbox in annos:
            assert len(_bbox) == 5, 'invalid bbox'

        classes = [int(_bbox[0]) for _bbox in annos]
        bboxes = [list(map(float, _bbox[1:])) for _bbox in annos]
        
        blob = {}
        blob['im_file'] = items[0]
        blob['gt_class'] = np.array(classes).astype(np.int32).reshape(-1, 1)
        blob['gt_bbox'] = np.array(bboxes).astype(np.float32).reshape(-1, 4)

        return blob