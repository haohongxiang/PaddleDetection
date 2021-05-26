import os
import numpy as np
from ppdet.core.workspace import register, serializable
from .dataset import DetDataset


@register
@serializable
class JsonDataSet(DetDataset):
    def __init__(self, dataset_dir=None,):
        print(dataset_dir)
        
    def parse_dataset(self):
        pass
    