from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import matplotlib.pyplot as plt, numpy as np, os, torch, random, cv2, json
from torchvision import models
from torchvision.transforms import v2 as transforms
from classes import BaseModelHandler
from vanilla_classification_models import ClassificationModelHandler
from vanilla_regression_models import RegressionModelHandler

# img_counter = 31 # Debug global counter for images processed

# === HYBRID MODELS === #
# Class for intial preprocessing and corner detection using a ResNet-based heatmap model
class ResNetHeatmap(nn.Module):
    def __init__(self, num_corners=4):
        super(ResNetHeatmap, self).__init__()
        resnet = models.resnet34(weights='DEFAULT')
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, num_corners, 2, stride=2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x
    
# 
class HybridTask3ModelHandler(BaseModelHandler):
    def __init__(self, model_path, corner_model_path='modelo.pth'):
        self.corner_model_path = corner_model_path
        super().__init__(model_path)
        self.corner_model = self.load_corner_model()
        self.inner_handler = self.get_inner_handler()
        self.target_size = 800  # Size for warped chessboard

    def load_corner_model(self):
        model = ResNetHeatmap().to(self.device)
        if os.path.exists(self.corner_model_path):
            print(f"Loading model from {self.corner_model_path} on device {self.device}")
            model.load_state_dict(torch.load(self.corner_model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Corner model not found at {self.corner_model_path}")
        model.eval()
        return model

    # load the appropriate resnet50 model based on model_path
    def load_model(self):
        return None # No model to load here, handled by inner handler

    def get_inner_handler(self):
        if "regression" in self.model_path:
            return RegressionModelHandler(self.model_path)
        elif "classification" in self.model_path:
            return ClassificationModelHandler(self.model_path)
        else:
            raise ValueError(f"Unsupported model type for inner handler: {self.model_path}")

    def preprocess(self):
        corner_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((224, 224)),
            transforms.ToDtype(torch.float32, scale=True)
        ])

        def hybrid_transform(img):
            global img_counter

            # Convert to RGB if needed
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_size = img.shape[1], img.shape[0]  # (width, height)

            # Step 1: Prepare image for corner detection
            img_tensor = corner_transform(img).unsqueeze(0).to(self.device)

            # Step 2: Predict corners
            with torch.no_grad():
                pred_heatmaps = self.corner_model(img_tensor).cpu().squeeze(0)
            
            pred_points_hm = self.heatmaps_to_points(pred_heatmaps)
            pred_points_img = self.points_to_image_coordinates(pred_points_hm, 56, original_size)

            # Step 3: Warp the image
            warped_img = self.calculate_homography_warped(img, np.array(pred_points_img))

            # Debug output warped image
            # output_path = os.path.join("debug_results", f"warped_{img_counter}.jpg")
            # warped_bgr = cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(output_path, warped_bgr)
            # print(f"Saved warped image to {output_path}")
            # img_counter +=1

            # Step 4: Apply inner handler's preprocessing
            inner_preprocessing = transforms.Compose([
                transforms.ToImage(),
                transforms.Resize((224, 224)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            warped_tensor = inner_preprocessing(warped_img)
            return warped_tensor

        return hybrid_transform

    def predict_batch(self, batch):
        batch = batch.to(self.device)
        return self.inner_handler.predict_batch(batch)
    
    # AUX FUNCS
    def heatmaps_to_points(self, heatmaps):
        num_corners, _, W = heatmaps.shape
        points = []
        for i in range(num_corners):
            heatmap = heatmaps[i]
            _, idx = torch.max(heatmap.view(-1), dim=0)
            y, x = divmod(idx.item(), W)
            points.append((x, y))
        return points

    def points_to_image_coordinates(self, points, origin_size, target_size):
        img_w, img_h = target_size
        scale_x = img_w / origin_size
        scale_y = img_h / origin_size
        image_points = [(x * scale_x, y * scale_y) for (x, y) in points]
        return image_points

    def calculate_homography_warped(self, image_np, pred_points):
        target_corners = np.array([
            [0, 0],
            [self.target_size - 1, 0],
            [self.target_size - 1, self.target_size - 1],
            [0, self.target_size - 1],
        ], dtype=np.float32)
        H, _ = cv2.findHomography(pred_points, target_corners)
        warped_image = cv2.warpPerspective(image_np, H, (self.target_size, self.target_size))
        return warped_image
