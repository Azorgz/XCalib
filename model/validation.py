from ImagesCameras import ImageTensor
from misc.Mytypes import Batch


class ValidationModule:
    def __init__(self, cfg):
        self.mode = cfg.mode_fusion
        self.cmap = cfg.color_map_infrared

    def __call__(self, batch: Batch) -> ImageTensor:
        rgb, ir = ImageTensor(batch.projections[0]), ImageTensor(batch.projections[1]).GRAY().RGB(self.cmap)
        if self.mode == 'alpha_blending':
            return rgb * 0.5 + ir * 0.5
        else:
            return rgb.combine(ir, method=self.mode)
