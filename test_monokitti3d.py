import os
import glob
from PIL import Image
from PIL import ImageDraw
import numpy as np
import copy
import math

from ppdet.core.workspace import load_config
from ppdet.core.workspace import create


cfg = load_config('./configs/mono_kitti3d/mono_kitti3d.yml')
mode = 'train'
worker_num = 1

dataset = cfg['TrainDataset']

train_loader = create('{}Reader'.format(mode.capitalize()))(dataset, worker_num)


# print(dataset[1])

for blob in train_loader:
    for k in blob:
        print(k, blob[k].shape)

def show(idx, dataset=train_loader.dataset):
    '''show
    '''
    blobs = dataset[idx]
    
    print('--------------------')
    print(type(blobs['image']))
    print('--------------------')
    
    
    image = Image.fromarray(blobs['image'])
    draw = ImageDraw.Draw(image)

    for bbx in blobs['bbox']:
        draw.rectangle(tuple(bbx), outline='red')
        x = (bbx[0::2]).sum() / 2
        y = (bbx[1::2]).sum() / 2
        draw.ellipse((x-3, y-3, x+3, y+3), fill='red')

#     pts = blobs['P2'][:3, :3] @ blobs['center3d']
#     for pt in pts.T:
#         x = pt[0] / pt[2]
#         y = pt[1] / pt[2]
#         draw.ellipse((x-3, y-3, x+3, y+3), fill='green')

    for pt in blobs['center2d']: 
        x = pt[0]
        y = pt[1]
        draw.ellipse((x-3, y-3, x+3, y+3), fill='green')

    image.save(f'test_{idx}.jpg')
    

# show(2)
# 
