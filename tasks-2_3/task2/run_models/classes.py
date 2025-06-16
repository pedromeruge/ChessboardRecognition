from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt, numpy as np, os, torch, random, cv2, json
from torchvision import models
from torchvision.transforms import v2 as transforms

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        return self.transform(img), path

# === ABSTRACT MODEL INTERFACE === #
class BaseModelHandler(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model = self.load_model()
        if (self.model != None):
            self.model = self.model.to(self.device)
            self.model.eval()


    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def predict_batch(self, batch):
        pass
