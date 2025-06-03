from .visualizer import Visualizer
from .visualizer_spatialtransformer import VisualizerSpatialTransformer, VisualizerSpatialTransformerCfg
from .visualizer_summary import VisualizerSummary, VisualizerSummaryCfg
from .visualizer_trajectory import VisualizerTrajectory, VisualizerTrajectoryCfg
from .visualizer_rig import VisualizerRig, VisualizerRigCfg

VISUALIZERS = {
    "summary": VisualizerSummary,
    "trajectory": VisualizerTrajectory,
    "rig": VisualizerRig,
    "spatialtransformer": VisualizerSpatialTransformer,
}

VisualizerCfg = VisualizerSummaryCfg | VisualizerTrajectoryCfg | VisualizerRigCfg | VisualizerSpatialTransformerCfg


def get_visualizers(cfgs: list[VisualizerCfg]) -> list[Visualizer]:
    return [VISUALIZERS[cfg.name](cfg) for cfg in cfgs]
