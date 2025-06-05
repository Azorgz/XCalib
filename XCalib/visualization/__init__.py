from .visualizer import Visualizer
from .visualizer_spatialtransformer import VisualizerSpatialTransformer, VisualizerSpatialTransformerCfg

VISUALIZERS = {"spatialtransformer": VisualizerSpatialTransformer}

VisualizerCfg = VisualizerSpatialTransformerCfg


def get_visualizers(cfgs: list[VisualizerCfg]) -> list[Visualizer]:
    return [VISUALIZERS[cfg.name](cfg) for cfg in cfgs]
