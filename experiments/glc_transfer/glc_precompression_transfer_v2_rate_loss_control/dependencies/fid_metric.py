import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision.models.inception import InceptionA, InceptionC, InceptionE


INCEPTION_WEIGHTS = Path("/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth")


class FIDInceptionA(InceptionA):
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_2(self.branch5x5_1(x))
        branch3x3dbl = self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x)))
        branch_pool = self.branch_pool(F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False))
        return torch.cat((branch1x1, branch5x5, branch3x3dbl, branch_pool), 1)


class FIDInceptionC(InceptionC):
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x)))
        branch7x7dbl = self.branch7x7dbl_5(self.branch7x7dbl_4(
            self.branch7x7dbl_3(self.branch7x7dbl_2(self.branch7x7dbl_1(x)))))
        branch_pool = self.branch_pool(F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False))
        return torch.cat((branch1x1, branch7x7, branch7x7dbl, branch_pool), 1)


class FIDInceptionE1(InceptionE):
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat((self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)), 1)
        branch3x3dbl = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        branch3x3dbl = torch.cat((self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)), 1)
        branch_pool = self.branch_pool(F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False))
        return torch.cat((branch1x1, branch3x3, branch3x3dbl, branch_pool), 1)


class FIDInceptionE2(FIDInceptionE1):
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat((self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)), 1)
        branch3x3dbl = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        branch3x3dbl = torch.cat((self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)), 1)
        branch_pool = self.branch_pool(F.max_pool2d(x, 3, stride=1, padding=1))
        return torch.cat((branch1x1, branch3x3, branch3x3dbl, branch_pool), 1)


class InceptionPool3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception_v3(init_weights=False, aux_logits=False, transform_input=False, num_classes=1008)
        self.model.Mixed_5b = FIDInceptionA(192, pool_features=32)
        self.model.Mixed_5c = FIDInceptionA(256, pool_features=64)
        self.model.Mixed_5d = FIDInceptionA(288, pool_features=64)
        self.model.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
        self.model.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
        self.model.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
        self.model.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
        self.model.Mixed_7b = FIDInceptionE1(1280)
        self.model.Mixed_7c = FIDInceptionE2(2048)
        self.model.load_state_dict(torch.load(INCEPTION_WEIGHTS, map_location="cpu"), strict=True)
        self.model.fc = torch.nn.Identity()
        self.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, images):
        # images are NCHW RGB in [0,1]. Match pytorch-fid input convention.
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        return self.model(images.mul(2).sub(1)).float()


@torch.inference_mode()
def extract_features(model, batches):
    values = []
    for batch in batches:
        values.append(model(batch).cpu().numpy())
    return np.concatenate(values, axis=0)


def low_rank_fid(first, second):
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    first_centered, second_centered = first - first.mean(0), second - second.mean(0)
    first_scale, second_scale = math.sqrt(len(first) - 1), math.sqrt(len(second) - 1)
    cross = (first_centered @ second_centered.T) / (first_scale * second_scale)
    covariance_trace = (np.square(first_centered).sum() / (len(first) - 1) +
                        np.square(second_centered).sum() / (len(second) - 1))
    mean_distance = np.square(first.mean(0) - second.mean(0)).sum()
    return float(max(mean_distance + covariance_trace - 2 * np.linalg.svd(cross, compute_uv=False).sum(), 0.0))
