from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import matplotlib.pyplot as plt, numpy as np, os, torch, random, cv2, json
from torchvision import models
from torchvision.transforms import v2 as transforms
from ultralytics import YOLO
from classes import * 

# === YOLO ADAPATION MODEL === #
class YoloCountingModelHandler(BaseModelHandler):
    def __init__(self, model_path):
        super().__init__(model_path)
        # self.img_counter = 31 # for debug
        self.img_size = 640  # YOLO input size
        self.model = self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLO model not found at {self.model_path}")
        model = YOLO(self.model_path)
        model.to(self.device)
        return model

    def preprocess(self):
        yolo_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToDtype(torch.float32, scale=True),
        ])

        def transform_fn(img, img_path=None):
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Apply transform
            img_tensor = yolo_transform(img).to(self.device)

            return img_tensor

        return transform_fn

    def predict_batch(self, batch):
        batch = batch.to(self.device)
        predictions = []
        with torch.no_grad():
            results = self.model(batch, verbose=False)
            for result in results:
                # Count number of bounding boxes
                num_boxes = len(result.boxes)
                predictions.append(num_boxes)

                # Debug: Save image with bounding boxes
                # img = result.orig_img
                # output_filename = f"yolo_input_{self.img_counter}.jpg"
                # output_path = os.path.join('debug_outputs', output_filename)
                # for box in result.boxes:
                #     x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                #     conf = box.conf[0].cpu().numpy()
                #     cls = int(box.cls[0].cpu().numpy())
                #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #     cv2.putText(img, f"cls:{cls} {conf:.2f}", (x1, y1 - 10),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.imwrite(output_path, img)
                # print(f"Saved YOLO output image with {num_boxes} boxes to {output_path}")
                # self.img_counter += 1

        return predictions
    
