import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



class ImageAnalyzer:
    
    def _hookedOutput(self, model, input, output):
        self.embeddings = output
        
    def __init__(self, dataset, class_names, model_path = None):
        self.dataset = dataset
        self.class_names = class_names
        # self.model = models.resnet18(pretrained = True)
        self.model = models.resnet34(pretrained = True)
        if model_path:
            num_classes = len(self.class_names)
            self.model = self._load_trained_model(self.model, model_path, num_classes)  

        layer = self.model._modules.get('avgpool')              # avgpool layer gives the embeddings needed
        self.model.eval()
        self.hook = layer.register_forward_hook(self._hookedOutput)
    
    def _load_trained_model(self, model, model_path, num_classes):
        # update fc layer with new in & output shapes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
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
    
    # TODO: make more generalizable 
    def imshow(self, img, labels = None):
        img = img.permute(1,2,0)   # move the color channels to last dimension
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406]) # unnormalize images
        img = np.clip(img, 0, 1)
        plt.figure(figsize=(16,16))
        plt.imshow(img)
        if labels is not None:
        # +1 so ImageLoader labels match dataset labels
            types = [self.dataset.classes_idx.get(str(i.item())) for i in labels+1]
            print('Type:    ', ' '.join(f'{str(types[i]):15s}' for i in range(len(types))))


    def cosine_similar_images(self, target_embedding, all_embeddings):
        cosine = nn.CosineSimilarity(dim=1) 
        cosine_similarity =  cosine(target_embedding, all_embeddings)
        # sort ascending -- the closer to 1, the more similar the images are
        return cosine_similarity.argsort(descending=True)[:10]

    def euclidean_similar_images(self, target_embeddings, all_embeddings):
        euclidean = (torch.pow(target_embeddings.cpu()-all_embeddings,2)).sum(dim=1).sqrt()
        # sort descending -- the smaller the distance, the more similar the images are
        return euclidean.argsort(descending = False)[:10]
    
    # TODO: make more generalizable 
    def show_best_results(self, top_results, dataset):
        top_images = torch.stack([self.dataset[i][0] for i in top_results])
        self.imshow(torchvision.utils.make_grid(top_images, nrow = 5, padding = 2))
        top_labels = [self.dataset[i][1] for i in top_results]
        # print("Type: ", ' '.join(f'{self.dataset.classes_idx.get(str(i+1)):15s}' for i in top_labels))