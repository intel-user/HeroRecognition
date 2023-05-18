import json


class Config:
    def __init__(self):
        self.device = "cpu"
        self.save_image = False
        self.use_onnx = False
        if not self.use_onnx:
            self.obj_det_checkpoint = "model/object_detection.pt"
        else:
            self.obj_det_checkpoint = "model/object_detection.onnx"
        self.feature_extractor = "model/preprocessor_config.json"
        self.vit_model_path = "model/embedding.pt"
        self.conf = 0.25
        self.iou = 0.5
        self.image_size = 224
        self.output_img = "result"
        self.log_file = "logs/app.log"
        self.storage = "storage"

    def __repr__(self):
        return json.dumps({key: getattr(self, key)
                           for key in self.__dir__() if "__" != key[:2] and "__" != key[-2:] and key != "dict"}
                          , indent=4)
