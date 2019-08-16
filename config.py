# -*- coding:utf-8 -*-

import torch
import warnings


class DefaultConfit(object):
    dataset_name = 'CUB'

    seed = 1
    batch_size = 50
    epochs = 100
    seed = 1
    latent_size = 64
    model_path = "./checkpoints/"+"checkpoint_cada_"+dataset_name+".pth"
    pretrained = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("warning：opt has not attribute %s" %k)
            setattr(self, k, v)

        print("user config.")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k + ':', getattr(self, k))

opt = DefaultConfit()
