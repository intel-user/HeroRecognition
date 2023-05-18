import logging
from typing import Dict, List

import numpy as np

from config.config import Config


class ModelInput:
    def __init__(self, image: np.ndarray, info: List[Dict] = None, name=None, config: Config = None):
        self.config = config
        self._image = image
        # [{'bounding_box': [],'confidence': tensor(0.2016), 'class_name': 'man'}]
        self._boxes_info = info
        self.chosen_object_id = 0
        self.left_object()
        self.champion_name = name
        self.logger = logging.getLogger(self.__class__.__name__)

    def left_object(self):
        list_x = [i["bounding_box"][0] for i in self._boxes_info]
        if list_x:
            self.chosen_object_id = list_x.index(min(list_x))
            self._boxes_info = [self._boxes_info[self.chosen_object_id]]

    def expand_box(self, alpha: float):
        h, w, _ = self._image.shape
        if self._boxes_info:
            for index in range(len(self._boxes_info)):
                boxes = self._boxes_info[index]["bounding_box"]
                top_left_x, top_left_y, bot_right_x, bot_right_y = boxes[0], boxes[1], boxes[2], boxes[3]
                top_left_x = int(max(top_left_x - alpha * (bot_right_x - top_left_x), 0))
                bot_right_x = int(min(bot_right_x + alpha * (bot_right_x - top_left_x), w))
                top_left_y = int(max(top_left_y - alpha * (bot_right_y - top_left_y), 0))
                bot_right_y = int(min(bot_right_y + alpha * (bot_right_y - top_left_y), h))
                self._boxes_info[index]["bounding_box"] = [top_left_x, top_left_y, bot_right_x, bot_right_y]

    def box_validate(self, box: List):
        if box[0] == box[2] or box[1] == box[3]:
            return False
        else:
            try:
                image = self._image[box[1]:box[3], box[0]:box[2]]
                shape = image.shape
                return True
            except Exception as e:
                return False

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image: np.ndarray):
        self._image = image

    @property
    def boxes_info(self):
        return self._boxes_info

    @boxes_info.setter
    def boxes_info(self, info: Dict):
        self._boxes_info = info

    def crop_image(self, index: int = None) -> np.ndarray:
        index = self.chosen_object_id if index is None else index
        box = self._boxes_info[index]["bounding_box"]
        return self._image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
