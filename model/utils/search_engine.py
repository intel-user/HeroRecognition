import glob
from typing import List
import logging

import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from transformers import ViTFeatureExtractor
import faiss

from config.config import Config
from object.data import ModelInput
from common.common import *
from object.singleton import Singleton


class Embedding:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(self.config.device)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.config.feature_extractor, size=224)
        self.load_pretrained_model()
        self.transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor()
        ])

    def load_pretrained_model(self):
        self.model = torch.load(self.config.vit_model_path, map_location=self.device)
        self.model.eval()

    def read_img(self, input_image: np.ndarray) -> torch.Tensor:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv2.resize(input_image, (self.config.image_size, self.config.image_size)))
        image = self.transform(image)
        return image

    def inference(self, images: List) -> List[List]:
        cloth_model_img = [self.read_img(input_image=i) for i in images]
        images = self.feature_extractor(images=cloth_model_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model.forward_one(images).tolist()

    def extract_image_batch(self, lst_data_model: List[ModelInput]) -> List[ModelInput]:
        for data_model in lst_data_model:
            batch_img = [data_model.crop_image(index=i) for i in range(len(data_model.boxes_info))]
            features = self.inference(images=batch_img)
            boxes_info = data_model.boxes_info
            for index, feature in enumerate(features):
                boxes_info[index][FEATURE] = np.array(feature).reshape(1, -1)
            data_model.boxes_info = boxes_info
            # self.logger.info(f"Inference Success: num of boxes: {len(data_model.boxes_info)}")
        self.logger.info(f"Inference Success: data length: {len(lst_data_model)}")
        return lst_data_model


class Faiss(metaclass=Singleton):
    def __init__(self, config: Config = None):
        super(Faiss, self).__init__()
        self.config = config
        self.index = faiss.IndexFlatL2(768)

    def search(self, query: np.ndarray, top_k: int = 1):
        return self.index.search(query, k=top_k)
