from ImagesCameras import ImageTensor
from torch import nn

from model.XCalib import XCalib
from options.options import get_options


def get_model(device, opt, **kwargs):
    cfg = get_options()
    cfg.run_parameters['mode'] = 'all_in_one'
    cfg.model['target'] = 0 if opt.direction == 'ir2vis' else 1
    cfg.model['depth_source'] = 0 if opt.direction == 'ir2vis' else 1
    model = XCalib(cfg=cfg).to(cfg.model['device'])

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.direction = opt.direction
            self.model = model.eval().to(device)

        def forward(self, img_vis, img_ir):
            out = ImageTensor(self.model.forward(img_vis, img_ir))
            return (out, img_ir) if self.direction == 'vis2ir' else (img_vis, out)
        
        def fit_to_data(self, data):
            cfg_new = get_options(data=data)
            cfg_new.model['target'] = 0 if self.direction == 'ir2vis' else 1
            cfg_new.model['depth_source'] = 0 if self.direction == 'ir2vis' else 1
            xcalib_new = XCalib(cfg=cfg_new).to(cfg_new.model['device'])
            self.model = xcalib_new
            if not self.model.optimized:
                self.model.optimize_parameters()

    return Model()
