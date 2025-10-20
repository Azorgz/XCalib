import os

import torch

from model.enhancer.swinir_arch import Resnet


def get_enhancer():
    enhancer = Resnet()
    checkpoint = torch.load(os.getcwd() + "/model/enhancer/checkpoint/enhancer.pth")
    enhancer.load_state_dict(checkpoint["params"])
    return enhancer.eval()
