import os
from pathlib import Path

import torch
from .models.cross_raft import CrossRAFT

THIS_DIR = Path(__file__).resolve().parent

def get_model_raft():
    model = CrossRAFT(adapter=True)
    state_dict = torch.load(str(THIS_DIR) + '/cross_raft_ckpt/model/checkpoint-10000.ckpt',
                            weights_only=True, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda().eval()
    return model