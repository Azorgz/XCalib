import os

import torch
from .models.cross_raft import CrossRAFT

def get_model_raft():
    model = CrossRAFT(adapter=True)
    state_dict = torch.load(os.getcwd() + '/model/flow/CrossModalFlow/cross_raft_ckpt/model/checkpoint-10000.ckpt', weights_only=True)['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda().eval()
    return model