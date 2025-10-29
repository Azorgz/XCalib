from .backbone import Backbone
from .backbone_depth_anything import BackboneZoe, BackboneZoeCfg
from .backbone_depth_pro import BackboneDepthPro, BackboneDepthProCfg

BACKBONES = {"zoe": BackboneZoe,
             'pro': BackboneDepthPro}

BackboneCfg = {"zoe": BackboneZoeCfg,
               'pro': BackboneDepthProCfg}


def get_backbone(
        cfg: BackboneCfg,
) -> Backbone:
    depth_model = BACKBONES[cfg.name](cfg)
    return depth_model
