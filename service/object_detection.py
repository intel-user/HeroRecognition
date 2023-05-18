import math
import os.path
from typing import List, Tuple
import logging

from PIL import Image

import cv2
import onnxruntime
import numpy as np
from glob import glob
from ultralytics import YOLO

from config.config import Config
from service.utils import xywh2xyxy, nms
from common.common import *


class ObjectDetection:
    def __init__(self, config: Config = None):
        self.config = config
        self.conf = self.config.conf
        self.iou = self.config.iou
        self.logger = logging.getLogger(__class__.__name__)
        if self.config.device in ['cpu', 'CPU', '-1', None, '']:
            self.device = "cpu"
            providers = ["CPUExecutionProvider"]
        else:
            self.device = f"cuda:{self.config.device}"
            providers = ["CUDAExecutionProvider"]

        if not self.config.use_onnx:
            self.model = YOLO(self.config.obj_det_checkpoint)
        else:
            self.model = onnxruntime.InferenceSession(self.config.obj_det_checkpoint, providers=providers)
            self.get_input_details()
            self.get_output_details()
        self.input_names = None
        self.input_shape = None
        self.input_width = None
        self.input_height = None
        self.output_names = None
        self.img_height = None
        self.img_width = None

    def get_input_details(self):
        model_inputs = self.model.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.model.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def batch_predict(self, source: str, is_storage=False):
        list_img, list_name = self.read_img(source, is_storage=is_storage)
        output = []
        for idx, img in enumerate(list_img):
            pred = None
            if img is not None:
                pred = self.detect_single_image(cv2_img=img)
            output.append({IMAGE: img, PRED: pred, NAME: list_name[idx]})
        return output

    def detect_single_image(self, cv2_img):
        origin_img = cv2_img.copy()
        final_result = []
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        h0, w0 = cv2_img.shape[:2]
        r = 416 / h0
        if r != 1:
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            cv2_img = cv2.resize(cv2_img, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)

        if not self.config.use_onnx:
            results = self.model.predict(source=cv2_img, conf=self.conf, device=self.device)
            xyxy_list = results[0].boxes.xyxy.tolist()
            conf_list = results[0].boxes.conf.tolist()
            cls_id_list = results[0].boxes.cls.tolist()
        else:
            input_tensor = self.prepare_input(cv2_img)
            outputs = self.inference(input_tensor)
            xyxy_list, conf_list, cls_id_list = self.process_output(outputs)

        for (idx, (xyxy, conf, cls_id)) in enumerate(zip(xyxy_list, conf_list, cls_id_list)):
            x1 = float(xyxy[0] / r)
            y1 = float(xyxy[1] / r)
            x2 = float(xyxy[2] / r)
            y2 = float(xyxy[3] / r)
            conf = float(conf)
            box = [x1, y1, x2, y2]
            final_result.append({BOUNDING_BOX: box, CATEGORY: [CLASS_NAME], CONFIDENCE: conf})
            if self.config.save_image:
                origin_img = cv2.rectangle(origin_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                cv2.putText(origin_img, CLASS_NAME, (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 0, 0), 2)
        if self.config.save_image:
            img_pil = Image.fromarray(origin_img)
            image_name = len(glob(f'{self.config.output_img}/*.jpg'))
            img_pil.save(f'{self.config.output_img}/{image_name}.jpg')
        return final_result

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensor):
        outputs = self.model.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf, :]
        scores = scores[scores > self.conf]
        if len(scores) == 0:
            return [], [], []
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions)
        indices = nms(boxes, scores, self.iou)
        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def read_img(self, source: str, is_storage=False) -> Tuple[List, List]:
        list_img = []
        list_name = []
        if os.path.isfile(source):
            img = self.imread(source)
            list_img.append(img)
        elif os.path.isdir(source):
            for root, dirs, files in os.walk(f"{source}", topdown=False):
                for file in files:
                    img = self.imread(os.path.join(root, file))
                    list_img.append(img)
                    if is_storage:
                        list_name.append(file.split("_")[0])
        if not list_name:
            list_name = [None]*len(list_img)
        return list_img, list_name

    def imread(self, img):
        try:
            return cv2.imread(img)
        except Exception as e:
            self.logger.error(f"Load Image Fail: {img}\nError: {e}")
            return None
