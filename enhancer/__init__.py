import os

import torch

from enhancer.swinir_arch import Resnet


def get_enhancer():
    enhancer = Resnet()
    checkpoint = torch.load(os.getcwd() + "/enhancer/checkpoint/enhancer.pth")
    enhancer.load_state_dict(checkpoint["params"])
    return enhancer.eval()
