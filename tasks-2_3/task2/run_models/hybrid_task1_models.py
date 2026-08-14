from abc import ABC
import torch
import cv2
import numpy as np
import os
from torchvision.transforms import v2 as transforms
from classes import BaseModelHandler
from vanilla_classification_models import ClassificationModelHandler
from vanilla_regression_models import RegressionModelHandler
from torchvision import models
import torch.nn as nn
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'task_1_code'))

from task_1_main import separate_horse_pipeline, pipeline_iter
from IO.json_handler import read_single_image

# === HYBRID MODELS === #
# Class for intial preprocessing and corner detection using traditional methods from task1

class HybridTask1ModelHandler(BaseModelHandler):
    def __init__(self, model_path):
        self.model_path = model_path
        self.inner_handler = self.get_inner_handler()
        super().__init__(model_path)
        self.img_counter = 31

        # load horse used in task1
        horse_path = "task_1_code/our_images/cavalinhoPequeno.jpg"
        self.horse_data = separate_horse_pipeline.apply(read_single_image(horse_path))[0]

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
        # inner handler preprocessing applied to image after warp, to feed to CNN
        inner_preprocessing = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((224, 224)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        def hybrid_transform(img):

            # traditional warping pipeline from task1
            warped_img = pipeline_iter(image=img, separate_horse_results=self.horse_data)

            # Debug output warped image
            # output_path = os.path.join("debug_outputs", f"warped_{self.img_counter}.jpg")
            # cv2.imwrite(output_path, warped_img)
            # print(f"Saved warped image to {output_path}")

            self.img_counter +=1

            # Apply inner preprocessing
            return inner_preprocessing(warped_img)

        return hybrid_transform

    def predict_batch(self, batch):
        batch = batch.to(self.device)
        return self.inner_handler.predict_batch(batch)