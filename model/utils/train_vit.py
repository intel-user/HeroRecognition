import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor

from config.config import Config

config_ = Config()


class Model(nn.Module):
    def __init__(self, config: Config = None):
        super(Model, self).__init__()
        self.vit_16 = ViTModel.from_pretrained(config.feature_extractor)
        self.cosine = nn.CosineSimilarity()

    def forward_one(self, x):
        x = self.vit_16(**x).pooler_output
        # x = self.vit_16(**x).last_hidden_state[:, 0]
        return x

    def forward(self, pos_img, neg_img, mix_img):
        pos_feature = self.forward_one(pos_img)
        neg_feature = self.forward_one(neg_img)
        mix_feature = self.forward_one(mix_img)
        pos_diff = self.cosine(pos_feature, mix_feature)
        neg_diff = self.cosine(neg_feature, mix_feature)
        return pos_diff, neg_diff
