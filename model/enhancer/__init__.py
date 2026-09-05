import os

import torch

from model.enhancer.swinir_arch import Resnet

ROOT = os.path.dirname(__file__)

def get_enhancer():
    enhancer = Resnet()
    checkpoint = torch.load(ROOT + "/checkpoint/enhancer_s.pth")
    enhancer.load_state_dict(checkpoint, strict=False)
    return enhancer.eval()
