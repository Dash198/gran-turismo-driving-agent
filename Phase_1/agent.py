import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class CNN(nn.Module):
    def __init__(self,
                 img_dims = (112, 112),
                 N_mean = [0.485, 0.456, 0.406],
                 N_std = [0.229, 0.224, 0.225]):

        super().__init__()
        self.preprocess = transforms.Compose([
            transforms.Resize(img_dims),
            transforms.ToTensor(),
            transforms.Normalize(mean=N_mean, std=N_std)
        ])

        base_model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()

    def forward(self, frames):
        tensors = []
        for frame in frames:
            img = Image.fromarray((frame*255).astype(np.uint8))
            tensor = self.preprocess(img)
            tensors.append(tensor)
        
        batch = torch.stack(tensors)

        with torch.no_grad():
            feats = self.model(batch)
            feats = feats.view(feats.size(0), -1)
            embedding = feats.mean(dim=0, keepdim=True)
        
        return embedding

class RandomAction:
    def __init__(self, N):
        self.n = N

    def select_action(self, X):
        return np.array([np.random.uniform(-1,1) for _ in range(self.n)])

class Agent:
    def __init__(self,
                 img_dims = (112, 112),
                 N_mean = [0.485, 0.456, 0.406],
                 N_std = [0.229, 0.224, 0.225]):
        
        self.cnn = CNN(img_dims=img_dims, N_mean=N_mean, N_std=N_std)
        self.policy = RandomAction(2)
    
    def select_action(self, frames):
        embedding = self.cnn(frames)
        return self.policy.select_action(embedding)