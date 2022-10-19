from __future__ import print_function
from __future__ import division

from torchvision import transforms
from torch.utils.data import dataset

import os
from collections import OrderedDict
import re
from PIL import Image

class RegexLabelExtractor():
    def __init__(self, pattern):
        self.pattern = pattern
        self._names = []
    
    def __call__(self, iterable):
        return [re.findall(self.pattern, value)[0] for value in iterable]
    
    

    
class LabelManager():
    def __init__(self, labels):
        self._label_to_idx = OrderedDict()    
        for label in labels:
            if label not in self._label_to_idx:
                self._label_to_idx[label] = len(self._label_to_idx)
        self._idx_to_label = {v:k for k,v in self._label_to_idx.items()}
    
    @property
    def keys(self):
        return list(self._label_to_idx.keys())
    
    def id_for_label(self, label):
        return self._label_to_idx[label]
    
    def label_for_id(self, idx):
        return self._idx_to_label[idx]
    
    def __len__(self):
        return len(self._label_to_idx)
    
    
    
class PetsDataset(dataset.Dataset):
    def __init__(self, data, tfms=None):
        super(PetsDataset, self).__init__()
        self.data = data
        self.transforms = tfms
    
    def __getitem__(self, idx):
        X = Image.open(self.data[idx][0])
        if X.mode != 'RGB':
            X = X.convert('RGB')
        y = self.data[idx][1]
        if self.transforms:
            X = self.transforms(X)
        return (X, y)
    
    def __len__(self):
        return len(self.data)

    
    
class DatasetManager():
    
    def __init__(self, base_dir, paths, label_extractor, tfms=None, seed=None):
        self._labels = label_extractor(paths)
        self.tfms = transforms.Compose([transforms.Resize((224,224)),               # resize image
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),    # change to Tensor
                                    transforms.Normalize(mean =  [0.485, 0.456, 0.406], 
                                                         std = [0.229, 0.224, 0.225])  ])
        self._label_manager = LabelManager(self._labels)
        self._label_ids = [self.label_manager.id_for_label(label) for label in self._labels]

        self.abs_paths = [os.path.join(base_dir, path) for path in paths]
        # self.train_data, self.valid_data = Splitter(valid_pct=valid_pct, seed=seed)(list(zip(self.abs_paths, self._label_ids)))
        self.data = list(zip(self.abs_paths, self._label_ids))
        
    @property
    def label_manager(self):
        return self._label_manager
    

    @property
    def dataset(self):    
        return PetsDataset(self.data, tfms=self.tfms)
    