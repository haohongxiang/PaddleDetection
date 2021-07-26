import os
import glob
from PIL import Image
from PIL import ImageDraw
import numpy as np
import copy
import math

import paddle

from ppdet.core.workspace import load_config
from ppdet.core.workspace import create


cfg = load_config('./configs/mono_kitti3d/mono_kitti3d.yml')
mode = 'train'
worker_num = 1

dataset = cfg['TrainDataset']

train_loader = create('{}Reader'.format(mode.capitalize()))(dataset, worker_num)


print(dataset[1])

for i, blob in enumerate(train_loader):
    for k, v in blob.items():
        if isinstance(v, paddle.Tensor):
            print(k, blob[k].shape)
        else:
            print(k, len(blob[k]))
            
    if i == 3:
        break
        
        
def show(idx, dataset=train_loader.dataset):
    '''show
    '''
    blobs = dataset[idx]
    
    print('--------------------')
    print(type(blobs['image']), blobs['image'].shape)
    print('--------------------')
    
    image = Image.fromarray(blobs['image'])
    draw = ImageDraw.Draw(image)

    for bbx in blobs['bbox']:
        draw.rectangle(tuple(bbx), outline='red')
        x = (bbx[0::2]).sum() / 2
        y = (bbx[1::2]).sum() / 2
        draw.ellipse((x-3, y-3, x+3, y+3), fill='red')
        
        print(bbx)
        
    for bbx in blobs['bbox_2d']:
        draw.rectangle(tuple(bbx), outline='blue')
        # x = (bbx[0::2]).sum() / 2
        # y = (bbx[1::2]).sum() / 2
        # draw.ellipse((x-3, y-3, x+3, y+3), fill='blue')

    for pt in blobs['center_2d']: 
        x = pt[0]
        y = pt[1]
        draw.ellipse((x-3, y-3, x+3, y+3), fill='green')

    image.save(f'test_{idx}.jpg')
    

show(23)
