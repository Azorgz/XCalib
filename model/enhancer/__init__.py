import os

import torch

from model.enhancer.swinir_arch import Resnet


def get_enhancer():
    enhancer = Resnet()
    checkpoint = torch.load(os.getcwd() + "/model/enhancer/checkpoint/enhancer_s.pth")
    enhancer.load_state_dict(checkpoint, strict=False)
    return enhancer.eval()
