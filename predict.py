import argparse
import glob
import logging
import os.path

from config.config import Config

from service.object_detection import ObjectDetection
from model.utils.search_engine import Embedding, Faiss
from utils.utils import setup_logging
from object.data import ModelInput
from common.common import *


class Inference:
    def __init__(self, config: Config = None, args=None):
        self.object_detection = ObjectDetection(config=config)
        self.embedding = Embedding(config=config)
        self.faiss = Faiss(config=config)
        self.map_key = []
        self.args = args
        self.config = config
        self.logger = logging.getLogger(__class__.__name__)

    def load_storage(self):
        output = self.object_detection.batch_predict(source=self.config.storage, is_storage=True)
        for pred in output:
            if pred[PRED]:
                lst_data_model = [ModelInput(image=pred[IMAGE], info=pred[PRED], name=pred[NAME], config=self.config)]
                lst_data_model = self.embedding.extract_image_batch(lst_data_model)
                for bbox in lst_data_model[0].boxes_info:
                    self.faiss.index.add(bbox[FEATURE])
                    self.map_key.append(pred[NAME])
        self.logger.info(f"Load Database Successful")

    def predict(self):
        output = self.object_detection.batch_predict(source=self.args.source)
        result = []
        for pred in output:
            if pred[PRED]:
                lst_data_model = [ModelInput(image=pred[IMAGE], info=pred[PRED], name=pred[NAME], config=self.config)]
                lst_data_model = self.embedding.extract_image_batch(lst_data_model)
                dis, index = self.faiss.search(lst_data_model[0].boxes_info[0][FEATURE])
                result.append(self.map_key[index[0][0]])
            else:
                result.append("Unknow")
        self.export(result=result)
        self.logger.info(f"Inference Successful")

    def export(self, result):
        files = []
        if os.path.isfile(self.args.source):
            files = [self.args.source]
        elif os.path.isdir(self.args.source):
            files = glob.glob(f"{self.args.source}/*")
        with open(f"{self.config.output_img}/result.txt", "w") as f:
            for idx, file in enumerate(files):
                f.write(f"{file}\t{result[idx]}\n")


def option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str,
                        default=r"C:\Users\tienhn\Desktop\Eklipse\Assignment\test_data\test_images",
                        help="Source image")
    return parser.parse_args()


if __name__ == "__main__":
    config_ = Config()
    setup_logging()
    argument = option()
    pipeline = Inference(config_, args=argument)
    pipeline.load_storage()
    pipeline.predict()
