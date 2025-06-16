from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import matplotlib.pyplot as plt, numpy as np, os, torch, random, cv2, json
from torchvision import models
from torchvision.transforms import v2 as transforms
from classes import BaseModelHandler

# === REGRESSION MODELS === #
class RegressionModelHandler(BaseModelHandler):
    def load_model(self):
        model=models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 1)
        print(f"Loading model from {self.model_path} on device {self.device}")
        model.load_state_dict(torch.load(self.model_path, map_location=self.device)['model'])
        return model

    def preprocess(self):
        return transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)), # fixed size crop to image, since board isn't near borders usually, and there are missleading pieces in the sides
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_batch(self, batch):
        batch = batch.to(self.device)
        with torch.set_grad_enabled(False):
            outputs = self.model(batch).squeeze()
            preds = outputs.detach().cpu().numpy() if outputs.ndim > 0 else np.array([outputs.item()])
            rounded = np.round(preds).astype(int)
            clamped = np.clip(rounded, 0, 32)
            return clamped.tolist()