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

    # if cfg.name == 'zoe':
    #     from .backbone_depth_anything import BackboneZoe, BackboneZoeCfg as BACKBONES, cfg
    # elif cfg.name == 'pro':
    #     from .backbone_depth_pro import BackboneDepthPro, BackboneDepthProCfg as BACKBONES, cfg
    # else:
    #     raise ValueError(f"Backbone {cfg.name} is not supported. Choose among: {', '.join(list(BackboneCfg.keys()))}")
    #
    # depth_model = BACKBONES(cfg)
