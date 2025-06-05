from .backbone import Backbone
from .backbone_depth_anything import BackboneZoe, BackboneZoeCfg
from .backbone_depth_anythingV2 import BackboneDepthAnythingV2, BackboneDepthAnythingV2Cfg
from .backbone_depth_pro import BackboneProCfg, BackbonePro


BACKBONES = {"zoe": BackboneZoe,
             "depthAnythingV2": BackboneDepthAnythingV2,
             "pro": BackbonePro}

BackboneCfg = BackboneZoeCfg | BackboneDepthAnythingV2Cfg | BackboneProCfg


def get_backbone(
        cfg: BackboneCfg,
) -> Backbone:
    return BACKBONES[cfg.name](cfg)
