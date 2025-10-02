from XCalib2.model.spatial_transformer import SpatialTransformerCfg, SpatialTransformer

SPATIAL_TRANSFORMER = {"spatialtransformer": SpatialTransformer}


def get_spatial_transformer(cfg: SpatialTransformerCfg) -> SpatialTransformer:
    return SPATIAL_TRANSFORMER[cfg.name](cfg)
