from torch import nn
from torch.nn import Conv2d


class depthRefiner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.preprocessor = nn.Sequential(
            Conv2d(4, 64, 3, dilation=3, padding=3, norm=None, act=None))
        self.featcompressor = nn.Sequential(Conv2d(channel * 2, channel * 2, 3, padding=1),
                                            Conv2d(channel * 2, channel, 3, padding=1, norm=None, act=None))
        oc = channel
        ic = channel + 2
        dilation = 1
        estimator = nn.ModuleList([])
        for i in range(depth - 1):
            oc = oc // 2
            estimator.append(
                Conv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=nn.BatchNorm2d))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc, 2, kernel_size=3, padding=1, dilation=1, act=None, norm=None))
        self.estimator = nn.Sequential(*estimator)

    def forward(self, depths, images):
        # Implement the depth refinement logic here
        refined_depths = depths  # Placeholder for actual refinement logic
        return refined_depths