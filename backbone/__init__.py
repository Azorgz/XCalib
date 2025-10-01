from .backbone import Backbone
from .backbone_depth_anything import BackboneZoe, BackboneZoeCfg


BACKBONES = {"zoe": BackboneZoe}

BackboneCfg = BackboneZoeCfg


def get_backbone(
        cfg: BackboneCfg,
) -> Backbone:
    depth_model = BACKBONES[cfg.name](cfg)
    depth_model.eval()
    return depth_model
