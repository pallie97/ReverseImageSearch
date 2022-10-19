# this is a distilled version of my class from Exploration/analyze_image.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class EmbeddingCalculator:

    def _hookedOutput(self, model, input, output):
        self.embeddings = output
        
    def __init__(self, model_path = None, num_classes=1000):
        self.num_classes = num_classes
        self.model = models.resnet34(pretrained = True)
        if model_path:
            self.model = self._load_trained_model(self.model, model_path, num_classes)  

        layer = self.model._modules.get('avgpool')              # avgpool layer gives the embeddings needed
        self.model.eval()
        self.hook = layer.register_forward_hook(self._hookedOutput)
    
    def _load_trained_model(self, model, model_path, num_classes):
        # update fc layer with new in & output shapes
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        # update the weights with those from the trained model
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        return model

    def transform(self, image):
        transformation = transforms.Compose([transforms.Resize((224,224)),               # resize image
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),    # change to Tensor
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                       std=[0.229, 0.224, 0.225]) ]) # normalize with mean & std from docs

        return transformation(image).unsqueeze(0)

    def getEmbeddings(self, input):
        with torch.no_grad():                     # use no_grad so that grad_fn doesn't output
            self.model(input)
        return self.embeddings.squeeze(-1).squeeze(-1)  # squeeze until we get a tensor of [1,512]