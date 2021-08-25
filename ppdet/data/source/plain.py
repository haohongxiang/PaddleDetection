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
import time
import pandas as pd
import pickle
import itertools

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
                 empty_ratio=1.,
                 use_cache=True,
                 cache_file='/paddle/workspace/dataset/cache.pkl'):
        
        super(PlainDetDataSet, self).__init__(dataset_dir, image_dir, anno_path,
                                          data_fields, sample_num)
        self.load_image_only = False
        self.load_semantic = False
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio
        
        self.num_worker = 32
        self.dataset_dir = dataset_dir
        self.anno_path = anno_path
        
        self.use_cache = use_cache
        self.cache_file = cache_file
        
        self.parse_dataset()

    
    def check_or_download_dataset(self, ):
        pass
    
    
    def parse_dataset(self, ):
        tic = time.time()
        self.parse_csv(self.use_cache, self.cache_file)
        
        print('\n loading done... ', time.time() - tic )
    
    
    def parse_csv(self, use_cache, cache_path):
        '''parse csv
        '''        
        if use_cache and os.path.exists(cache_path):
            self.roidbs = pickle.load(open(cache_path, 'rb'))
            return 
        
        if not isinstance(self.anno_path, (list, tuple)):
            anno_paths = (self.anno_path, )
        else:
            anno_paths = self.anno_path
            
        anno_paths = [os.path.join(self.dataset_dir, anno) for anno in anno_paths]
        
        eps = 1e-5
        def _parse(path):
            def _format(group):
                '''anno_format'''
                # gt_bbox = group[['XMin', 'YMin', 'XMax', 'YMax']].to_numpy()
                gt_bbox = np.array([group[n].to_list() for n in ['XMin', 'YMin', 'XMax', 'YMax']]).T
                gt_class = group.LabelName.to_list()
                im_file = group.ImagePath.to_list()[0]

                anno = {
                    'im_file': im_file,
                    'gt_bbox': gt_bbox,
                    'gt_class': gt_class,
                }
                
                return anno
        
            data = pd.read_csv(path)
            print('before: ', data.shape, )
            
            data = data.dropna()
            data = data[(data.XMax - data.XMin > eps) & (data.YMax - data.YMin > eps)]
            # _msk = [os.path.exists(os.path.join(self.dataset_dir, fstr)) for fstr in data['ImagePath']]
            _msk = data['ImagePath'].apply(lambda x: os.path.exists(os.path.join(self.dataset_dir, x)))
            data = data[_msk]
            print('after filter: ', data.shape, )

            data = data.groupby('ImagePath').apply(_format).reset_index(name='anno')
            print('after groupby: ', data.shape)

            return data.anno.to_list()
        
        roidbs = [_parse(path) for path in anno_paths]
            
        self.roidbs = list(itertools.chain(*roidbs))

        if use_cache and not os.path.exists(cache_path):
            pickle.dump(self.roidbs, open(cache_path, 'wb'))

        
        
        
        
    def parse_txt(self):
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
        
        logger.warning('loading...')
        tic = time.time()
        
        with futures.ThreadPoolExecutor(self.num_worker) as executor:
            roidbs = executor.map(self._parse_line, lines)
        
        self.roidbs = [t for t in roidbs if t]
        
        print('total time: ', time.time() - tic)
        logger.warning('loading data done...')
        

    def _parse_line(self, lin):
        '''
        im_file.jpg, class_id x1 y1 x2 y2, class_id x1 y1 x2 y2
        '''
        items = lin.strip(' ,\t').split(',')
        im_path = os.path.join(self.dataset_dir, items[0].strip())
                
        if not os.path.exists(im_path):
            logger.warning(f'{im_path} not exist...')
            return None

        if len(items) == 1:
            logger.warning(f'{items} empty..')
            return None

        annos = [_bbox.strip().split('\t') for _bbox in items[1:]]
        for _bbox in annos:
            assert len(_bbox) == 5, f'invalid bbox, {_bbox}'

        classes = [int(_bbox[0]) for _bbox in annos]
        bboxes = [list(map(float, _bbox[1:])) for _bbox in annos]
        
        blob = {}
        blob['im_file'] = im_path
        blob['gt_class'] = np.array(classes).astype(np.int32).reshape(-1, 1)
        blob['gt_bbox'] = np.array(bboxes).astype(np.float32).reshape(-1, 4)

        return blob
    
    
        
    def _sample_empty(self, records, num):
        # if empty_ratio is out of [0. ,1.), do not sample the records
        if self.empty_ratio < 0. or self.empty_ratio >= 1.:
            return records
        import random
        sample_num = int(num * self.empty_ratio / (1 - self.empty_ratio))
        records = random.sample(records, sample_num)
        return records

    
    
    
    
    


names = ['person', 
        'bicycle', 
        'car', 
        'motorcycle', 
        'airplane', 
        'bus', 
        'train', 
        'truck', 
        'boat', 
        'traffic light', 
        'fire hydrant', 
        'stop sign', 
        'parking meter', 
        'bench', 
        'bird', 
        'cat', 
        'dog', 
        'horse', 
        'sheep', 
        'cow', 
        'elephant', 
        'bear', 
        'zebra', 
        'giraffe', 
        'backpack', 
        'umbrella', 
        'handbag', 
        'tie', 
        'suitcase', 
        'frisbee', 
        'skis', 
        'snowboard', 
        'sports ball', 
        'kite', 
        'baseball bat', 
        'baseball glove', 
        'skateboard', 
        'surfboard', 
        'tennis racket', 
        'bottle', 
        'wine glass', 
        'cup', 
        'fork', 
        'knife', 
        'spoon', 
        'bowl', 
        'banana', 
        'apple', 
        'sandwich', 
        'orange', 
        'broccoli', 
        'carrot', 
        'hot dog', 
        'pizza', 
        'donut', 
        'cake', 
        'chair', 
        'couch', 
        'potted plant', 
        'bed', 
        'dining table', 
        'toilet', 
        'tv', 
        'laptop', 
        'mouse', 
        'remote', 
        'keyboard', 
        'cell phone', 
        'microwave', 
        'oven', 
        'toaster', 
        'sink', 
        'refrigerator', 
        'book', 
        'clock', 
        'vase', 
        'scissors', 
        'teddy bear', 
        'hair drier', 
        'toothbrush',

        # obj365
        'grapefruit',
        'tomato',
        'power outlet',
        'lipstick',
        'bakset',
        'napkin',
        'slide',
        'antelope',
        'radiator',
        'lighter',
        'ship',
        'egg tart',
        'billards',
        'bowl/basin',
        'duck',
        'hammer',
        'rice',
        'hamburger',
        'folder',
        'side table',
        'chips',
        'crane',
        'fire truck',
        'monkey',
        'chopsticks',
        'jellyfish',
        'mirror',
        'table tennis ',
        'cigar/cigarette ',
        'necklace',
        'pencil case',
        'table teniis paddle',
        'soccer',
        'notepaper',
        'bracelet',
        'kettle',
        'other shoes',
        'cello',
        'potato',
        'flask',
        'violin',
        'paddle',
        'dolphin',
        'toilet paper',
        'spring rolls',
        'fire extinguisher',
        'broom',
        'dessert',
        'sneakers',
        'green onion',
        'ladder',
        'awning',
        'baozi',
        'curling',
        'hamimelon',
        'brush',
        'boots',
        'grape',
        'traffic sign',
        'guitar',
        'yak',
        'mop',
        'trophy',
        'buttefly',
        'lemon',
        'corn',
        'hotair ballon',
        'pepper',
        'coffee machine',
        'toiletry',
        'heavy truck',
        'storage box',
        'sandals',
        'eggplant',
        'penguin',
        'avocado',
        'blackboard/whiteboard',
        'stapler',
        'kiwi fruit',
        'barbell',
        'dishwasher',
        'cd',
        'plate',
        'cherry',
        'dumpling',
        'tape',
        'microphone',
        'stuffed toy',
        'mask',
        'urinal',
        'other fish',
        'towel',
        'plum',
        'soap',
        'wheelchair',
        'eraser',
        'drum',
        'street lights',
        'recorder',
        'cutting/chopping board',
        'american football',
        'green vegetables',
        'medal',
        'swing',
        'megaphone',
        'stroller',
        'garlic',
        'traffic cone',
        'lantern',
        'tricycle',
        'extention cord',
        'cheese',
        'high heels',
        'saxophone',
        'slippers',
        'gas stove',
        'tablet',
        'hanger',
        'radish',
        'tuba',
        'oyster',
        'ice cream',
        'dinning table',
        'washing machine/drying machine',
        'crosswalk sign',
        'noddles',
        'rickshaw',
        'electric drill',
        'earphone',
        'lifesaver',
        'handbag/satchel',
        'lion',
        'cosmetics mirror',
        'sports car',
        'candle',
        'baseball',
        'golf club',
        'cleaning products',
        'crab',
        'telephone',
        'chicken',
        'extractor',
        'stool',
        'ring',
        'flower',
        'tent',
        'board eraser',
        'showerhead',
        'other balls',
        'lettuce',
        'faucet',
        'belt',
        'pear',
        'cabbage',
        'hockey stick',
        'lamp',
        'donkey',
        'head phone',
        'gun',
        'air conditioner',
        'piano',
        'onion',
        'french',
        'surveillance camera',
        'asparagus',
        'machinery vehicle',
        'durian',
        'pineapple',
        'target',
        'sushi',
        'watch',
        'skiboard',
        'green beans',
        'speed limit sign',
        'candy',
        'sausage',
        'bread',
        'comb',
        'watermelon',
        'steak',
        'tea pot',
        'wild bird',
        'pliers',
        'peach',
        'printer',
        'sailboat',
        'gloves',
        'carriage',
        'cosmetics',
        'induction cooker',
        'red cabbage',
        'okra',
        'scale',
        'fishing rod',
        'hair dryer',
        'briefcase',
        'converter',
        'trolley',
        'helmet',
        'deer',
        'treadmill',
        'pickup truck',
        'cucumber',
        'suv',
        'jug',
        'tissue',
        'game board',
        'binoculars',
        'poker card',
        'pasta',
        'rice cooker',
        'volleyball',
        'tong',
        'strawberry',
        'goldfish',
        'pen/pencil',
        'router/modem',
        'pigeon',
        'skating and skiing shoes',
        'coconut',
        'canned',
        'paint brush',
        'egg',
        'picture/frame',
        'screwdriver',
        'cabinet/shelf',
        'tripod',
        'coffee table',
        'desk',
        'papaya',
        'globe',
        'moniter/tv',
        'bathtub',
        'nightstand',
        'hat',
        'meat ball',
        'leather shoes',
        'wallet/purse',
        'flag',
        'trombone',
        'dumbbell',
        'cue',
        'shovel',
        'flute',
        'computer box',
        'trumpet',
        'scooter',
        'nuts',
        'pie',
        'helicopter',
        'blender',
        'fan',
        'lobster',
        'pumpkin',
        'basketball',
        'hurdle',
        'campel',
        'carpet',
        'tennis',
        'luggage',
        'mushroon',
        'golf ball',
        'bow tie',
        'orange/tangerine',
        'seal',
        'barrel/bucket',
        'pig',
        'pomegranate',
        'cymbal',
        'cookies',
        'pillow',
        'hoverboard',
        'formula 1 ',
        'swan',
        'goose',
        'scallop',
        'projector',
        'calculator',
        'french fries',
        'pot',
        'rabbit',
        'tape measur/ ruler',
        'trash bin can',
        'speaker',
        'parrot',
        'camera',
        'chainsaw',
        'key',
        'shrimp',
        'marker',
        'ballon',
        'van',
        'mango',
        'glasses',
        'cosmetics brush/eyeliner pencil',
        'ambulance',

        # oid
        'serving tray',
        'fax',
        'carnivore',
        'window blind',
        'swim cap',
        'doughnut',
        'tick',
        'human nose',
        'shelf',
        'measuring cup',
        'curtain',
        'snowman',
        'canary',
        'zucchini',
        'fashion accessory',
        'weapon',
        'accordion',
        'house',
        'dinosaur',
        'waffle iron',
        'pastry',
        'volleyball (ball)',
        'poster',
        'building',
        'wrench',
        'cooking spray',
        'human eye',
        'boot',
        'doll',
        'kitchen utensil',
        'lynx',
        'marine invertebrates',
        'envelope',
        'tart',
        'cabinetry',
        'bronze sculpture',
        'pressure cooker',
        'coat',
        'jacuzzi',
        'fruit',
        'aircraft',
        'harmonica',
        'beer',
        'caterpillar',
        'sewing machine',
        'dice',
        'christmas tree',
        'canoe',
        'skull',
        'scarf',
        'jaguar (animal)',
        'sword',
        'barrel',
        'ring binder',
        'cream',
        'sock',
        'golf cart',
        'tiara',
        'cocktail shaker',
        'squid',
        'porch',
        'paper cutter',
        'plumbing fixture',
        'drinking straw',
        'squirrel',
        'maple',
        'gondola',
        'container',
        'human mouth',
        'paper towel',
        'chest of drawers',
        'sea turtle',
        'trousers',
        'frog',
        'torch',
        'miniskirt',
        'table tennis racket',
        'woman',
        'submarine',
        'common sunflower',
        'computer monitor',
        'light bulb',
        'cart',
        'billiard table',
        'chime',
        'maracas',
        'tower',
        'ostrich',
        'koala',
        'chisel',
        'rhinoceros',
        'shirt',
        'tablet computer',
        'otter',
        'taxi',
        'football',
        'cricket ball',
        'infant bed',
        'limousine',
        'bookcase',
        'bomb',
        'tableware',
        'fox',
        'earrings',
        'magpie',
        'fireplace',
        'ipod',
        'bull',
        'snowmobile',
        'indoor rower',
        'alpaca',
        'red panda',
        'hiking equipment',
        'sofa bed',
        'computer keyboard',
        'soap dispenser',
        'wall clock',
        'burrito',
        'sombrero',
        'football helmet',
        'muffin',
        'vehicle',
        'digital clock',
        'hippopotamus',
        'ladybug',
        'bagel',
        'stationary bicycle',
        'hand dryer',
        'bathroom accessory',
        'skyscraper',
        'tire',
        'goat',
        'washing machine',
        'heater',
        'sunglasses',
        'missile',
        'human hand',
        'food processor',
        'swimming pool',
        'luggage and bags',
        'human leg',
        'nail (construction)',
        'musical instrument',
        'tortoise',
        'mixing bowl',
        'food',
        'cocktail',
        'balance beam',
        'tea',
        'stethoscope',
        'vegetable',
        'slow cooker',
        'butterfly',
        'mobile phone',
        'cassette deck',
        'rugby ball',
        'invertebrate',
        'syringe',
        'fedora',
        'wok',
        'rays and skates',
        'fish',
        'dress',
        'mule',
        'training bench',
        'teapot',
        'seahorse',
        'rocket',
        'salt and pepper shakers',
        'flying disc',
        'lavender (plant)',
        'wine rack',
        'houseplant',
        'beehive',
        'beetle',
        'kitchen knife',
        'ruler',
        'reptile',
        'human head',
        'shorts',
        'coffeemaker',
        'turtle',
        'face powder',
        'whisk',
        'rose',
        'racket',
        'box',
        'power plugs and sockets',
        'bidet',
        'bottle opener',
        'girl',
        'bathroom cabinet',
        'raccoon',
        'dog bed',
        'axe',
        'mammal',
        'ant',
        'skirt',
        'palm tree',
        'coin',
        'lizard',
        'waffle',
        'isopod',
        'shotgun',
        'home appliance',
        'jeans',
        'cupboard',
        'tin can',
        'worm',
        'flowerpot',
        'stretcher',
        'tank',
        'scoreboard',
        'crown',
        'ratchet (device)',
        'pitcher (container)',
        'filing cabinet',
        'sculpture',
        'parachute',
        'juice',
        'humidifier',
        'barge',
        'crutch',
        'artichoke',
        'sandal',
        'sparrow',
        'shower',
        'spider',
        'harp',
        'human arm',
        'dairy product',
        'pizza cutter',
        'adhesive tape',
        'waste container',
        'pretzel',
        'personal care',
        'cheetah',
        'bee',
        'alarm clock',
        'snowplow',
        'insect',
        'polar bear',
        'guacamole',
        'balloon',
        'sun hat',
        'studio couch',
        'human hair',
        'medical equipment',
        'mushroom',
        'oboe',
        'sea lion',
        'kitchen appliance',
        'croissant',
        'banjo',
        'jet ski',
        'cannon',
        'saucer',
        'bicycle helmet',
        'computer mouse',
        'hedgehog',
        'coffee cup',
        'crocodile',
        'handgun',
        'bell pepper',
        'wheel',
        'bat (animal)',
        'tool',
        'milk',
        'closet',
        'band-aid',
        'hair spray',
        'marine mammal',
        'watercraft',
        'bust',
        'salad',
        'ski',
        'eagle',
        'land vehicle',
        'lily',
        'tiger',
        'perfume',
        'human face',
        'drill (tool)',
        'door',
        'hamster',
        'shark',
        'brassiere',
        'footwear',
        'rifle',
        'horizontal bar',
        'office supplies',
        'shellfish',
        'human beard',
        'cake stand',
        'flashlight',
        'taco',
        'lighthouse',
        'whale',
        'wardrobe',
        'roller skates',
        'headphones',
        'human body',
        'french horn',
        'remote control',
        'garden asparagus',
        'human foot',
        'stairs',
        'armadillo',
        'wood-burning stove',
        'seafood',
        'pancake',
        'spice rack',
        'snack',
        'tree',
        'spatula',
        'ceiling fan',
        'squash (plant)',
        'swimwear',
        'kitchen & dining room table',
        'television',
        'cutting board',
        'jacket',
        'personal flotation device',
        'man',
        'sports uniform',
        'ladle',
        'castle',
        'furniture',
        'clothing',
        'drink',
        'suit',
        'light switch',
        'kangaroo',
        'harbor seal',
        'honeycomb',
        'drawer',
        'convenience store',
        'sports equipment',
        'kitchenware',
        'street light',
        'fountain',
        'grinder',
        'winter melon',
        'baked goods',
        'cantaloupe',
        'tree house',
        'seat belt',
        'beaker',
        'microwave oven',
        'coffee',
        'moths and butterflies',
        'unicycle',
        'brown bear',
        'cowboy hat',
        'cookie',
        'dragonfly',
        'billboard',
        'goggles',
        'egg (food)',
        'mixer',
        'loveseat',
        'fast food',
        'punching bag',
        'glove',
        'can opener',
        'panda',
        'musical keyboard',
        'turkey',
        'window',
        'mechanical fan',
        'skunk',
        'popcorn',
        'scorpion',
        'raven',
        'bicycle wheel',
        'picnic basket',
        'dagger',
        'bowling equipment',
        'owl',
        'facial tissue holder',
        'plant',
        'vehicle registration plate',
        'starfish',
        'auto part',
        'submarine sandwich',
        'table',
        'picture frame',
        'wine',
        'cattle',
        'door handle',
        'tennis ball',
        'frying pan',
        'common fig',
        'boy',
        'plastic bag',
        'diaper',
        'harpsichord',
        'platter',
        'falcon',
        'countertop',
        'pencil sharpener',
        'whiteboard',
        'camel',
        'toy',
        'animal',
        'centipede',
        'organ (musical instrument)',
        'blue jay',
        'segway',
        'snake',
        'leopard',
        'willow',
        'mug',
        'ball',
        'human ear',
        'tap',
        'woodpecker',
        'corded phone',
        'cat furniture',
        'snail',
        'office building',
        'pen',
        'porcupine',
        'bow and arrow']

names_map = dict(zip(names, range(len(names))))
