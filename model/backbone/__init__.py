from .backbone import Backbone
from .backbone_depth_anything_v2 import BackboneDepthAnything, BackboneDepthAnythingCfg
from .backbone_zoedepth import BackboneZoe, BackboneZoeCfg
from .backbone_depth_pro import BackboneDepthPro, BackboneDepthProCfg
from .backbone_yolo import BackboneYolo, BackboneYoloCfg

BACKBONES = {"zoe": BackboneZoe,
             'pro': BackboneDepthPro,
             'yolo': BackboneYolo,
             'depth_anything': BackboneDepthAnything}

BackboneCfg = {"zoe": BackboneZoeCfg,
               'pro': BackboneDepthProCfg,
               'yolo': BackboneYoloCfg,
               'depth_anything': BackboneDepthAnythingCfg}


def get_backbone(
        cfg: BackboneCfg,
) -> Backbone:
    depth_model = BACKBONES[cfg.name](cfg)
    return depth_model
