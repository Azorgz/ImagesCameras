import os
from typing import Optional, Union, Sequence, List, Any
from warnings import warn

import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import joint_bilateral_blur
from scipy.special import gamma
from torch import Tensor, nn, tensor
from torch.nn.functional import binary_cross_entropy, conv2d
from torch_similarity.modules import GradientCorrelationLoss2d
from torchmetrics import MeanSquaredError, Metric
from torchmetrics.clustering import MutualInfoScore, NormalizedMutualInfoScore
from torchmetrics.functional.image import visual_information_fidelity
from torchmetrics.functional.image.scc import _scc_update, _scc_per_channel_compute
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchvision import models
from torchvision.transforms.v2.functional import gaussian_blur

from ..Image import ImageTensor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
######################### METRIC ##############################################
from ..tools.gradient_tools import grad_tensor

EPS = 1e-6


class BaseMetric(Metric):
    ##
    # A class defining the general basic metric working with Tensor on GPU
    # Set to True if the metric is differentiable else set to False
    is_differentiable: Optional[bool] = True

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, device: torch.device = None, **kwargs):
        super().__init__()
        self.preds = None
        self.target = None
        self.target2 = None
        self.memorize_past_input = False
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("target2", default=[], dist_reduce_fx="cat")
        self.metric = "Base Metric"
        self.mask = None
        self.weights = None
        self.value = 0
        self.range_min = 0
        self.range_max = 1
        self.commentary = "Just a base"
        # self.ratio_list = torch.tensor([1, 3 / 4, 2 / 3, 9 / 16, 9 / 21])
        # self.ratio_dict = {1: [512, 512],
        #                    round(3 / 4, 3): [480, 640],
        #                    round(2 / 3, 3): [448, 672],
        #                    round(9 / 16, 3): [405, 720],
        #                    round(9 / 21, 3): [340, 800]}
        self.to(device)

    def update(self, *args, mask=None, weights=None, **kwargs) -> None:
        assert len(
            args) >= self.min_arg, f"At least {self.min_arg} arguments are required to update the metric {self.metric}, but only {len(args)} were given"
        if len(args) == 1:
            preds = args[0]
            target2, target = None, None
        elif len(args) == 2:
            if self.max_arg == 1:
                print(f"Warning: More than 1 argument was given to the metric {self.metric}, "
                      f"only the first will be used as input")
                preds = args[0]
                target2, target = None, None
            else:
                target, preds = args[:2]
                target2 = None
        else:
            if self.max_arg == 1:
                warn(f"More than 1 argument was given to the metric {self.metric},"
                     f"only the first will be used as input")
                preds = args[0]
                target2, target = None, None
            elif self.max_arg == 2:
                warn(f"More than 2 arguments were given to the metric {self.metric},"
                     f"only the first two will be used as target and preds")
                target, preds = args[:2]
                target2 = None
            elif len(args) > self.max_arg:
                warn(f"More than 3 arguments were given to the metric {self.metric},"
                     f"only the first two will be used as target and preds")
                target, target2, preds = args[:3]
            else:
                target, target2, preds = args[:3]

        preds = ImageTensor(preds)
        target = ImageTensor(target) if target is not None else None
        target2 = ImageTensor(target2) if target2 is not None else None
        image_true = None
        image_true_2 = None
        if target is not None:
            if target.channel_num == preds.channel_num:
                image_test = preds
                image_true = target
                if target2 is not None:
                    image_true_2 = target2
            elif target.channel_num > 1:
                image_true = ImageTensor(target.mean(dim=target.channel_pos, keepdim=True), batched=target.batched)
                image_test = preds
                if target2 is not None:
                    image_true_2 = ImageTensor(target2.mean(dim=target2.channel_pos, keepdim=True),
                                               batched=target2.batched)
            else:
                image_true = target
                image_test = ImageTensor(preds.mean(dim=preds.channel_pos, keepdim=True), batched=preds.batched)
                if target2 is not None:
                    if target2.channel_num == target.channel_num:
                        image_true_2 = target2
                    else:
                        image_true_2 = ImageTensor(target2.mean(dim=target2.channel_pos, keepdim=True),
                                                   batched=target2.batched)
        else:
            image_test = preds

        # size = self._determine_size_from_ratio(image_test)
        image_test = image_test.to_tensor()
        image_true = image_true.to_tensor() if image_true is not None else None
        image_true_2 = image_true_2.to_tensor() if image_true_2 is not None else None
        if mask is not None:
            mask = ImageTensor(mask * 1.)
            self.mask = mask.match_shape(image_test).to_tensor().to(torch.bool).to(self.device)
        else:
            self.mask = torch.ones_like(image_test, device=self.device).to(torch.bool)
        if weights is not None:
            weights = ImageTensor(weights / weights.max())
            self.weights = weights.match_shape(image_test).to_tensor().to(self.device)
        else:
            self.weights = torch.ones_like(image_test, device=self.device)

        self.preds.append(image_test)
        self.target.append(image_true)
        self.target2.append(image_true_2)

    def compute(self):
        im1 = self.preds[-1] if self.preds else None
        im2 = self.target[-1] if self.target else None
        im3 = self.target2[-1] if self.target2 else None
        if not self.memorize_past_input:
            self.preds = []
            self.target = []
            self.target2 = []
        return im1, im2, im3

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
             ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        return self._plot(val, ax)

    def _determine_size_from_ratio(self, image_true):
        ratio = image_true.image_size[0] / image_true.image_size[1]
        idx_ratio = round(float(self.ratio_list[torch.argmin((self.ratio_list - ratio) ** 2)]), 3)
        size = self.ratio_dict[float(idx_ratio)]
        return size

    def __add__(self, other):
        assert isinstance(other, BaseMetric)
        if self.metric == other.metric:
            self.value = self.value + other.value
        else:
            self.value = self.scale + other.scale
        return self

    @property
    def scale(self):
        return self.value

    def __str__(self):
        ##
        # Redefine the way of printing
        return f"{self.metric} metric : {self.value} | between {self.range_min} and {self.range_max} | {self.commentary}"


class SSIM(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    return_image: Optional[bool] = True

    def __init__(self, device: torch.device, abs_values: bool = False, no_negative_values: bool = False,
                 kernel_size: int = 11):
        super().__init__(device)
        self.ssim = StructuralSimilarityIndexMeasure(gaussian_kernel=True,
                                                     sigma=1.5,
                                                     kernel_size=kernel_size,
                                                     reduction=None,
                                                     data_range=None,
                                                     k1=0.01, k2=0.03,
                                                     return_full_image=True,
                                                     return_contrast_sensitivity=False).to(self.device)
        self.metric = "SSIM"
        self.commentary = "The higher, the better"
        self.abs_values = abs_values
        self.no_negative_values = no_negative_values

    def update(self, *args, mask=None, return_image=False, **kwargs) -> None:
        super().update(*args, mask=mask, **kwargs)
        self.return_image = return_image

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        _, image = self.ssim(image_test, image_true)
        _, image_2 = self.ssim(image_test, image_true_2) if image_true_2 is not None else (None, None)
        image = torch.abs(image)
        if image_2 is not None:
            image = (image + torch.abs(image_2)) / 2
        self.value = image[:, :, self.mask[0, 0, :, :]].mean(dim=[-1, -2]).squeeze()
        self.mask = None
        self.ssim.reset()
        del _
        # self.value = self.ssim(self.image_test * mask, self.image_true * mask)
        if self.return_image:
            if self.abs_values:
                return ImageTensor(torch.abs(image.mean(dim=1, keepdim=True)), permute_image=True).RGB('gray')
            elif self.no_negative_values:
                return ImageTensor(torch.clamp(image.mean(dim=1, keepdim=True), min=0), permute_image=True).RGB('gray')
            else:
                return ImageTensor(image.mean(dim=1, keepdim=True), permute_image=True).RGB('gray')
        else:
            return self.value

    def scale(self):
        self.range_max += self.range_max
        return self


class MultiScaleSSIM(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.ms_ssim = MS_SSIM(gaussian_kernel=True,
                               sigma=1.5,
                               kernel_size=11,
                               reduction=None,
                               data_range=None,
                               k1=0.01, k2=0.03,
                               betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)).to(self.device)
        self.metric = "Multi Scale SSIM"
        self.commentary = "The higher, the better"

    def update(self, *args, mask=None, **kwargs) -> None:
        super().update(*args, mask=mask, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        value = self.ms_ssim(image_test * self.mask * self.weights,
                             image_true * self.mask * self.weights)
        if image_true_2 is not None:
            value_2 = self.ms_ssim(image_test * self.mask * self.weights,
                                   image_true_2 * self.mask * self.weights)
            value = (value + value_2) / 2
        self.value = value
        nb_pixel_im = image_test.shape[-2] * image_test.shape[-1]
        nb_pixel_mask = (~self.mask[0, 0]).to(torch.float32).sum()
        self.mask = None
        # Remove the perfect SSIM given by the mask
        self.value = (self.value * nb_pixel_im - nb_pixel_mask) / ((nb_pixel_im - nb_pixel_mask) or 1)
        self.ms_ssim.reset()
        return self.value

    def scale(self):
        return self.value


class MSE(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    return_image: Optional[bool] = True

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "MSE"
        self.range_max = 1
        self.commentary = "The lower, the better"

    def update(self, *args, mask=None, return_image=False, **kwargs) -> None:
        super().update(*args, mask=mask, **kwargs)
        self.return_image = return_image

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        image_mse = (image_test - image_true) ** 2 * (self.mask * 1.)
        if image_true_2 is not None:
            image_mse_2 = (image_test - image_true_2) ** 2 * (self.mask * 1.)
            image_mse = (image_mse + image_mse_2) / 2
        self.value = torch.mean(image_mse, dim=(1, 2, 3))
        self.reset()
        if self.return_image:
            return ImageTensor(image_mse.mean(dim=1, keepdim=True)).RGB('gray')
        return self.value

    def scale(self):
        self.range_max = 2
        return self.value


class RMSE(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    return_image: Optional[bool] = True

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.rmse = MeanSquaredError(squared=False).to(self.device)
        self.metric = "RMSE"
        self.range_max = 1

    def update(self, *args, mask=None, return_image=False, **kwargs) -> None:
        super().update(*args, mask=mask, **kwargs)
        self.return_image = return_image

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        diff = (image_test - image_true) * (self.mask * 1.)
        image_mse = torch.abs(diff)
        if image_true_2 is not None:
            diff_2 = (image_test - image_true_2) * (self.mask * 1.)
            image_mse_2 = torch.abs(diff_2)
            image_mse = (image_mse + image_mse_2) / 2
        self.value = torch.sqrt(torch.mean(image_mse ** 2, dim=(1, 2, 3)) + EPS)
        self.reset()
        if self.return_image:
            return ImageTensor(image_mse.mean(dim=1, keepdim=True)).RGB('gray')
        return self.value

    def scale(self):
        self.value = super().scale
        return self.value


class MI(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    is_differentiable = True
    higher_is_better = True
    full_state_update = False
    plot_lower_bound: float = 0.0
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Mutual Information Score"
        self.range_max = 1
        self.commentary = "The higher, the better"
        self.mi = MutualInfoScore().to(device)

    def update(self, *args, mask=None, **kwargs) -> None:
        super().update(*args, mask=mask, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        image_test = (image_test * self.mask * 255).flatten(1).to(torch.uint8)
        image_true = (image_true * self.mask * 255).flatten(1).to(torch.uint8)
        value = torch.stack([self.mi(img_true, img_test) for img_true, img_test in zip(image_true, image_test)], dim=0)
        if image_true_2 is not None:
            image_true_2 = (image_true_2 * self.mask * 255).flatten(1).to(torch.uint8)
            value_2 = torch.stack(
                [self.mi(img_true2, img_test) for img_true2, img_test in zip(image_true_2, image_test)], dim=0)
            value = (value + value_2) / 2
        self.value = value
        self.reset()
        return self.value

    def scale(self):
        self.range_max = 2
        return self.value


class nMI(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    is_differentiable = True
    higher_is_better = True
    full_state_update = False
    plot_lower_bound: float = 0.0
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Normalized Mutual Information Score"
        self.range_max = 1
        self.commentary = "The higher, the better"
        self.nmi = NormalizedMutualInfoScore('arithmetic').to(device)

    def update(self, *args, mask=None, **kwargs) -> None:
        super().update(*args, mask=mask, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        image_test = (image_test * self.mask * 255).flatten(1).to(torch.uint8)
        image_true = (image_true * self.mask * 255).flatten(1).to(torch.uint8)
        value = torch.stack([self.nmi(img_true, img_test) for img_true, img_test in zip(image_true, image_test)], dim=0)
        if image_true_2 is not None:
            image_true_2 = (image_true_2 * self.mask * 255).flatten(1).to(torch.uint8)
            value_2 = torch.stack(
                [self.nmi(img_true2, img_test) for img_true2, img_test in zip(image_true_2, image_test)], dim=0)
            value = (value + value_2) / 2
        self.value = value
        self.reset()
        return self.value

    def scale(self):
        self.range_max = 2
        return self.value


class PSNR(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.psnr = PeakSignalNoiseRatio(data_range=1., base=10.0, dim=(1, 2, 3), reduction=None).to(self.device)
        self.metric = "Peak Signal Noise Ratio"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = "inf"

    def update(self, *args, mask=None, **kwargs) -> None:
        super().update(*args, mask=mask, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        try:
            image_true = image_true * self.mask
            image_test = image_test * self.mask
            value = self.psnr(image_true, image_test)
            if image_true_2 is not None:
                image_true_2 = image_true_2 * self.mask
                value_2 = self.psnr(image_true_2, image_test)
                value = (value + value_2) / 2
            self.value = value
        except RuntimeError:
            self.value = -1

        self.psnr.reset()
        return self.value

    def __add__(self, other):
        assert isinstance(other, BaseMetric)
        if self.metric == other.metric:
            self.value = self.value + other.value * self.range_max / other.range_max
            self.range_max += self.range_max
        else:
            self.value = self.scale() + other.scale()
        return self


class NEC(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True
    is_differentiable = True
    full_state_update = False
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Edges Correlation"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1
        self.return_image = False
        self.return_coeff = False

    def update(self, *args, mask=None, weights=None, return_image=False, return_coeff=False, **kwargs) -> None:
        super().update(*args, mask=mask, weights=weights, **kwargs)
        self.return_image = return_image
        self.return_coeff = return_coeff

    @staticmethod
    def _filter_image(img1, img2):
        try:
            img1_filtered = joint_bilateral_blur(img1, img2, (3, 3), 0.1, (1.5, 1.5))
            img2_filtered = joint_bilateral_blur(img2, img1, (3, 3), 0.1, (1.5, 1.5))
            return img1_filtered, img2_filtered
        except torch.OutOfMemoryError:
            warn("Warning: Not enough memory to apply the joint bilateral filter, skipping it for this batch")
            return img1, img2

    def _compute_image_and_ref(self, img_true, img_test):
        ref_true = grad_tensor(ImageTensor(img_true, batched=img_true.shape[0] > 1, device=self.device)) * self.mask[:,
                                                                                                           :2]
        ref_test = grad_tensor(ImageTensor(img_test, batched=img_test.shape[0] > 1, device=self.device)) * self.mask[:,
                                                                                                           :2]
        dot_prod = torch.abs(torch.cos(ref_true[:, 1] - ref_test[:, 1])) * 2 - 1
        image_nec = ref_true[:, 0] * ref_test[:, 0] * dot_prod * self.weights[:, 0] * self.mask[:, 0]
        nec_ref = torch.sqrt(
            torch.abs(torch.sum(ref_true[:, 0] ** 2 * self.weights[:, 0] * self.mask[:, 0], dim=[-1, -2]) *
                      torch.sum(ref_test[:, 0] ** 2 * self.weights[:, 0] * self.mask[:, 0], dim=[-1, -2])) + EPS)
        return image_nec, nec_ref

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        # image_true, image_test = self._filter_image(image_true, image_test)
        image_nec, nec_ref = self._compute_image_and_ref(image_true, image_test)
        self.value = (image_nec.sum(dim=[-1, -2]) / nec_ref)

        if image_true_2 is not None:
            # image_true_2, image_test_2 = self._filter_image(image_true_2, image_test)
            image_nec_2, nec_ref_2 = self._compute_image_and_ref(image_true_2, image_test)
            image_nec = (image_nec + image_nec_2) / 2
            self.value = (self.value + (image_nec_2.sum(dim=[-1, -2]) / nec_ref_2)) / 2

        if self.return_image:
            return ImageTensor(image_nec, permute_image=True).RGB('gray')
        elif self.return_coeff:
            return self.value, nec_ref
        else:
            return self.value


class SCC(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True
    is_differentiable = True
    full_state_update = False
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device, high_pass_filter: Optional[Tensor] = None,
                 window_size: int = 8, **kwargs: Any):
        super().__init__(device)
        self.metric = "Spatial Correlation Coefficient"
        self.commentary = "The higher, the better"
        self.range_min = -1
        self.range_max = 1
        self.return_image = False
        self.return_coeff = False
        if high_pass_filter is None:
            high_pass_filter = tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).to(device)

        self.hp_filter = high_pass_filter
        self.ws = window_size

        self.add_state("scc_score", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")
        self.to(device)

    def update(self, *args,
               mask=None, weights=None, **kwargs) -> None:
        super().update(*args, mask=mask, weights=weights, **kwargs)

    def _compute_scc(self, preds, target):
        preds, target, hp_filter = _scc_update(preds, target, self.hp_filter, self.ws)
        scc_per_channel = [
            _scc_per_channel_compute(preds[:, i, :, :].unsqueeze(1), target[:, i, :, :].unsqueeze(1), hp_filter,
                                     self.ws)
            for i in range(preds.size(1))
        ]
        scc_score = torch.mean(torch.cat(scc_per_channel, dim=1), dim=[1, 2, 3])
        self.scc_score += scc_score.sum()
        self.total += preds.size(0)
        return scc_score

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        value = self._compute_scc(image_test * self.mask * self.weights, image_true * self.mask * self.weights)
        if image_true_2 is not None:
            value_2 = self._compute_scc(image_test * self.mask * self.weights, image_true_2 * self.mask * self.weights)
            value = (value + value_2) / 2
        self.value = value
        self.reset()
        return self.value

    def scale(self):
        self.range_max += self.range_max
        return self


class NCC(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True
    is_differentiable = True
    full_state_update = False
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Normalized Correlation Coefficient"
        self.commentary = "The higher, the better"
        self.range_min = - torch.sqrt(tensor(EPS))
        self.range_max = torch.sqrt(tensor(EPS))
        self.return_image = False
        self.return_coeff = False

    def update(self, *args, mask=None, weights=None, return_image=False, return_coeff=False, **kwargs) -> None:
        args_avg = [ImageTensor(arg.mean(dim=1, keepdim=True)) for arg in args[:3]]
        super().update(*args_avg, mask=mask, weights=weights, **kwargs)
        self.return_image = return_image
        self.return_coeff = return_coeff

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        image_test = (image_test - image_test.mean(dim=[1, 2, 3], keepdim=True)) * self.mask
        image_true = (image_true - image_true.mean(dim=[1, 2, 3], keepdim=True)) * self.mask
        num = torch.sum(image_test * image_true * self.weights, dim=[1, 2, 3])
        den = torch.sqrt(torch.sum(image_test ** 2 * self.weights, dim=[1, 2, 3]) *
                         torch.sum(image_true ** 2 * self.weights, dim=[1, 2, 3]) + EPS)
        value = num / den
        if image_true_2 is not None:
            image_true_2 = (image_true_2 - image_true_2.mean(dim=[1, 2, 3], keepdim=True)) * self.mask
            num_2 = torch.sum(image_test * image_true_2 * self.weights, dim=[1, 2, 3])
            den_2 = torch.sqrt(torch.sum(image_test ** 2 * self.weights, dim=[1, 2, 3]) *
                               torch.sum(image_true_2 ** 2 * self.weights, dim=[1, 2, 3]) + 1e-6)
            value_2 = num_2 / den_2
            value = (value + value_2) / 2
        self.value = value
        if self.return_image:
            return ImageTensor(image_test * image_true, permute_image=True).RGB('gray')
        elif self.return_coeff:
            return self.value, den
        else:
            return self.value

    def scale(self):
        self.range_max += self.range_max
        return self


class GradientCorrelation(BaseMetric, GradientCorrelationLoss2d):
    higher_is_better: Optional[bool] = True
    is_differentiable = True
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super(GradientCorrelation, self).__init__(return_map=True, device=device)
        self.metric = "Gradient Correlation"
        self.return_map = True
        self.return_image = False

    def _compute_map(self, x, y, mask=None, weights=None):
        _, gc_map = GradientCorrelationLoss2d.forward(self, x, y)

        if weights is not None:
            gc_map_ = gc_map * weights
        else:
            gc_map_ = gc_map
        if mask is not None:
            gc_map_ *= mask
        return gc_map_

    def update(self, *args, mask=None, weights=None, return_image=False, **kwargs) -> None:
        super().update(*args, mask=mask, weights=weights, **kwargs)
        self.return_image = return_image

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        res = self._compute_map(image_test, image_true,
                                mask=self.mask,
                                weights=self.weights)
        if image_true_2 is not None:
            res_2 = self._compute_map(image_test, image_true_2,
                                      mask=self.mask,
                                      weights=self.weights)
            res = (res + res_2) / 2
        value = res.flatten(1, -1).sum(-1)
        self.value = value
        if self.return_image:
            return res
        else:
            return self.value


class VGG(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = False
    is_differentiable = True
    full_state_update = False
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Semantic loss"
        self.commentary = "The lower, the better"
        self.range_min = 0
        self.range_max = 1
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        for param in vgg.parameters():
            param.requires_grad = False
        self.layers = nn.ModuleList([
            vgg[:4],  # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:18],  # relu3_4
            vgg[18:27],  # relu4_4
            vgg[27:36]  # relu5_4
        ]).to(device).eval()

        self.criterion = lambda x, y: nn.L1Loss(reduction='none')(x, y).mean(dim=[1, 2, 3])
        self.w = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        if image_test.shape[1] == 1:
            image_test = image_test.repeat(1, 3, 1, 1)
            image_test = (image_test - self.mean) / self.std
        if image_true.shape[1] == 1:
            image_true = image_true.repeat(1, 3, 1, 1)
            image_true = (image_true - self.mean) / self.std
        if image_true_2 is not None:
            if image_true_2.shape[1] == 1:
                image_true_2 = image_true_2.repeat(1, 3, 1, 1)
            image_true_2 = (image_true_2 - self.mean) / self.std
            feat_true_2 = image_true_2
        else:
            feat_true_2 = None
        value = 0
        feat_test = image_test
        feat_true = image_true
        for i, layer in enumerate(self.layers):
            feat_test = layer(feat_test)
            feat_true = layer(feat_true)
            if feat_true_2 is not None:
                feat_true_2 = layer(feat_true_2)
            value += (self.w[i] * (self.criterion(feat_true, feat_test) +
                                   (self.criterion(feat_true_2, feat_test) if feat_true_2 is not None else 0)) *
                      (0.5 if feat_true_2 is not None else 1))
        self.value = value
        return self.value

    def scale(self):
        self.range_max += self.range_max
        return self


class EN(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True
    is_differentiable = True
    full_state_update = False
    min_arg = 1
    max_arg = 1

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Entropy"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1 - torch.log(tensor(EPS))

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def compute(self):
        image_test, *_ = super().compute()
        hist = torch.stack([torch.histc(img.flatten(), bins=256, min=0., max=1.) for img in image_test], dim=0)
        hist /= (hist.sum(dim=1, keepdim=True) + EPS)
        self.value = -torch.sum(hist * torch.log(hist + EPS), dim=1)
        return self.value

    def scale(self):
        self.range_max += self.range_max
        return self


class CrossEntropy(BaseMetric):
    """Binary cross-entropy between fused image and the elementwise average of IR and VI after applying
        sigmoid/clamping. Returns BCE loss scalar."""
    higher_is_better: Optional[bool] = False
    is_differentiable = True
    full_state_update = False
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Binary cross-entropy"
        self.commentary = "The lower, the better"
        self.range_min = 0
        self.range_max = 1

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        image_true = torch.sigmoid(image_true)
        image_test = torch.sigmoid(image_test)
        if image_true_2 is not None:
            image_true_2 = torch.sigmoid(image_true_2)
            image_true = (image_true + image_true_2) / 2
        image_true = torch.clamp(image_true, EPS, 1.0 - EPS)
        image_test = torch.clamp(image_test, EPS, 1.0 - EPS)
        self.value = binary_cross_entropy(image_test, image_true, reduction='none')
        return self.value.mean(dim=[1, 2, 3])


class QNCIE(BaseMetric):
    """Quality via Normalized Cross-Information Entropy: normalizes images, computes pairwise normalized
    cross-correlations, computes eigenvalues of correlation matrix and returns an entropy-based score in [0,1]."""
    higher_is_better: Optional[bool] = True
    is_differentiable = True
    full_state_update = False
    min_arg = 3
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Binary cross-entropy"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1
        self.NCC = NCC(device)

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        image_test_n = self.normalize(image_test)
        image_true_n = self.normalize(image_true)
        image_true_2_n = self.normalize(image_true_2)

        NCCxy = self.NCC(image_true_n, image_true_2_n)
        NCCxf = self.NCC(image_true_n, image_test_n)
        NCCyf = self.NCC(image_true_2_n, image_test_n)
        one = torch.ones_like(NCCxy, device=image_test.device)
        R = torch.stack([torch.stack([one, NCCxy, NCCxf], dim=-1),
                         torch.stack([NCCxy, one, NCCyf], dim=-1),
                         torch.stack([NCCxf, NCCyf, one], dim=-1)], dim=-1)
        r = torch.linalg.eigvals(R).real
        K = 3
        HR = torch.sum((r * torch.log2(r / K) / K).view(image_test.shape[0], -1), dim=-1)
        HR = -HR / 8
        self.value = 1 - HR
        return self.value

    @staticmethod
    def normalize(img):
        B = img.shape[0]
        mini = img.view(B, -1).min(dim=-1)[0]
        maxi = img.view(B, -1).max(dim=-1)[0]
        return (img - mini.view(B, 1, 1, 1)) / (maxi.view(B, 1, 1, 1) - mini.view(B, 1, 1, 1) + EPS)

    @staticmethod
    def NCC(img1, img2):
        mean1 = torch.mean(img1)
        mean2 = torch.mean(img2)
        numerator = torch.sum((img1 - mean1) * (img2 - mean2), dim=[1, 2, 3])
        denominator = torch.sqrt(
            torch.sum((img1 - mean1) ** 2, dim=[1, 2, 3]) * torch.sum((img2 - mean2) ** 2, dim=[1, 2, 3]))
        return numerator / (denominator + EPS)

    def scale(self):
        self.range_max += self.range_max
        return self


class ShannonEntropy(BaseMetric):
    """Tsallis (or Shannon when q=1) entropies computed per image from histogram of size `ksize`."""
    higher_is_better: Optional[bool] = False
    is_differentiable = True
    full_state_update = False
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Binary cross-entropy"
        self.commentary = "The lower, the better"
        self.range_min = 0
        self.range_max = 1
        self.q = 1
        self.ksize = 256

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    @staticmethod
    def compute_entropy(img, q, ksize):
        img_tensor = img.view(img.shape[0], -1).float()
        histogram = torch.stack([torch.histc(img, bins=ksize, min=0, max=ksize - 1) for img in img_tensor], dim=0)
        probabilities = histogram / (torch.sum(histogram, dim=1, keepdim=True) + 1e-10)
        if q == 1:
            entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10), dim=1)
        else:
            entropy = (1 / (q - 1)) * (1 - torch.sum(probabilities ** q, dim=1))
        return entropy

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        TE = self.compute_entropy(image_true, self.q, self.ksize)
        TE_f = self.compute_entropy(image_test, self.q, self.ksize)
        if image_true_2 is not None:
            TE = (self.compute_entropy(image_true_2, self.q, self.ksize) + TE) / 2
        self.value = TE - TE_f
        return self.value


class EdgeIntensity(BaseMetric):
    """Edge intensity: applies Sobel filters, computes gradient magnitude and returns mean gradient magnitude (scalar)."""
    higher_is_better: Optional[bool] = True
    is_differentiable = True
    full_state_update = False
    min_arg = 1
    max_arg = 1

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Binary cross-entropy"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1
        self.sobel_kernel_x = torch.tensor([[-1., 0., 1.],
                                            [-2., 0., 2.],
                                            [-1., 0., 1.]]).to(device)

        self.sobel_kernel_y = torch.tensor([[-1., -2., -1.],
                                            [0., 0., 0.],
                                            [1., 2., 1.]]).to(device)

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def compute(self):
        image_test, *_ = super().compute()
        image_test = image_test.mean(dim=1, keepdim=True)
        sobel_kernel_x = self.sobel_kernel_x.view(1, 1, 3, 3)
        sobel_kernel_y = self.sobel_kernel_y.view(1, 1, 3, 3)
        gx = conv2d(image_test, sobel_kernel_x, padding=1)
        gy = conv2d(image_test, sobel_kernel_y, padding=1)

        g = torch.sqrt(gx ** 2 + gy ** 2)
        self.value = torch.sum(g, dim=[1, 2, 3]) / (image_test.shape[2] * image_test.shape[3] + EPS)
        return self.value


class SpatialFrequency(BaseMetric):
    """Spatial frequency: computes row and column frequencies and returns their root mean square (scalar)."""
    higher_is_better: Optional[bool] = True
    is_differentiable = True
    full_state_update = False
    min_arg = 1
    max_arg = 1

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Spatial Frequency loss"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def compute(self):
        image_test, *_ = super().compute()
        RF = torch.sqrt(torch.mean((image_test[:, :, 1:, :] - image_test[:, :, :-1, :]) ** 2, dim=[1, 2, 3]))
        CF = torch.sqrt(torch.mean((image_test[:, :, :, 1:] - image_test[:, :, :, :-1]) ** 2, dim=[1, 2, 3]))
        self.value = torch.sqrt(RF ** 2 + CF ** 2 + EPS)
        return self.value


class StandartDeviation(BaseMetric):
    """Standard deviation: computes standard deviation of pixel intensities (scalar)."""
    higher_is_better: Optional[bool] = True
    is_differentiable = True
    full_state_update = False
    min_arg = 1
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Standard Deviation"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def compute(self):
        image_test, image_true, image_true2 = super().compute()
        if image_true is not None and image_true2 is None:
            m = image_true.mean(dim=[1, 2, 3], keepdim=True)
            value = torch.sqrt(torch.mean((image_test - m) ** 2, dim=[1, 2, 3]) + EPS)
        elif image_true is None and image_true2 is None:
            value = torch.std(image_test, dim=[1, 2, 3])
        else:
            m1 = image_true.mean(dim=[1, 2, 3], keepdim=True)
            m2 = image_true2.mean(dim=[1, 2, 3], keepdim=True)
            value = (torch.mean((image_test - m1) ** 2, dim=[1, 2, 3]) + torch.mean((image_test - m2) ** 2,
                                                                                    dim=[1, 2, 3])) / 2
        self.value = value
        return self.value


class VIF(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True
    is_differentiable = True
    full_state_update = False
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Visual Information Fidelity"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1
        self.vif = visual_information_fidelity

    def update(self, *args, mask=None, **kwargs) -> None:
        super().update(*args, mask=mask, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        value = self.vif(image_true * self.mask, image_test * self.mask, reduction="none")
        #
        # value = self.vifp_mscale(image_true * self.mask * self.weights,
        #                   image_test * self.mask * self.weights)
        if image_true_2 is not None:
            # value_2 = self.vifp_mscale(image_true_2 * self.mask * self.weights,
            #                     image_test * self.mask * self.weights)
            value_2 = self.vif(image_true_2 * self.mask, image_test * self.mask, reduction="none")
            value = (value + value_2) / 2
        self.value = value
        self.reset()
        return self.value

    def vifp_mscale(self, ref, dist, eps=1e-10):
        """
        Multiscale VIF (torch implementation).
        Inputs:
            ref, dist : (B,1,H,W) or (B,H,W)
        Returns:
            scalar tensor
        """

        if ref.dim() == 3:
            ref = ref.unsqueeze(1)
            dist = dist.unsqueeze(1)

        sigma_nsq = 2.0
        num = 0.0
        den = 0.0

        B, C, H, W = ref.shape
        device = ref.device
        dtype = ref.dtype

        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            win = self.fspecial_gaussian(N, N / 5, device=device, dtype=dtype)

            padding = N // 2

            if scale > 1:
                ref = F.conv2d(ref, win, padding=padding)
                dist = F.conv2d(dist, win, padding=padding)
                ref = ref[:, :, ::2, ::2]
                dist = dist[:, :, ::2, ::2]

            mu1 = F.conv2d(ref, win, padding=padding)
            mu2 = F.conv2d(dist, win, padding=padding)

            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(ref * ref, win, padding=padding) - mu1_sq
            sigma2_sq = F.conv2d(dist * dist, win, padding=padding) - mu2_sq
            sigma12 = F.conv2d(ref * dist, win, padding=padding) - mu1_mu2

            sigma1_sq = torch.clamp(sigma1_sq, min=0)
            sigma2_sq = torch.clamp(sigma2_sq, min=0)

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            mask1 = sigma1_sq < eps
            mask2 = sigma2_sq < eps

            g = torch.where(mask1, torch.zeros_like(g), g)
            sv_sq = torch.where(mask1, sigma2_sq, sv_sq)
            sigma1_sq = torch.where(mask1, torch.zeros_like(sigma1_sq), sigma1_sq)

            g = torch.where(mask2, torch.zeros_like(g), g)
            sv_sq = torch.where(mask2, torch.zeros_like(sv_sq), sv_sq)

            negative_g = g < 0
            sv_sq = torch.where(negative_g, sigma2_sq, sv_sq)
            g = torch.where(negative_g, torch.zeros_like(g), g)

            sv_sq = torch.clamp(sv_sq, min=eps)

            num += torch.sum(torch.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)), dim=[1, 2, 3])
            den += torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq), dim=[1, 2, 3])

        vifp = num / (den + eps)
        return vifp

    @staticmethod
    def fspecial_gaussian(kernel_size, sigma, device=None, dtype=None):
        """
        Create 2D Gaussian kernel (torch version).
        Returns tensor of shape (1,1,k,k)
        """
        coords = torch.arange(kernel_size, device=device, dtype=dtype)
        coords -= (kernel_size - 1) / 2.

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = torch.outer(g, g)
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def scale(self):
        self.range_max += self.range_max
        return self.value


class CorrelationCoefficient(BaseMetric):
    """Correlation coefficient: computes Pearson correlation between A–F and B–F and returns their mean."""
    higher_is_better: Optional[bool] = True
    is_differentiable = True
    full_state_update = False
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Correlation Coefficient - Pearson correlation"
        self.commentary = "The higher, the better"
        self.range_min = -1
        self.range_max = 1

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        image_test = image_test - image_test.mean(dim=[1, 2, 3], keepdim=True)
        image_true = image_true - image_true.mean(dim=[1, 2, 3], keepdim=True)
        value = torch.sum(image_true * image_test, dim=[1, 2, 3]) / torch.sqrt(
            torch.sum(image_true ** 2, dim=[1, 2, 3]) *
            torch.sum(image_test ** 2, dim=[1, 2, 3]) + EPS)
        if image_true_2 is not None:
            image_true_2 = image_true_2 - image_true_2.mean(dim=[1, 2, 3], keepdim=True)
            value = torch.sum(image_true_2 * image_test, dim=[1, 2, 3]) / torch.sqrt(
                torch.sum(image_true_2 ** 2, dim=[1, 2, 3]) *
                torch.sum(image_test ** 2, dim=[1, 2, 3]) + EPS) / 2 + value / 2
        self.value = value
        return self.value


class StructuralCorrelationDifference(BaseMetric):
    """Structural correlation difference: computes Pearson correlation between A–F and B–F and returns their absolute difference."""
    higher_is_better: Optional[bool] = False
    is_differentiable = True
    full_state_update = False
    min_arg = 2
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Structural Correlation Difference - Absolute difference of Pearson correlations"
        self.commentary = "The lower, the better"
        self.range_min = -2
        self.range_max = 2

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        num1 = torch.sum(image_true * image_test, dim=[1, 2, 3])
        den1 = torch.sqrt(
            torch.sum(image_true ** 2, dim=[1, 2, 3]) *
            torch.sum(image_test ** 2, dim=[1, 2, 3])
        ).clamp_min(EPS)

        corr1 = num1 / den1

        if image_true_2 is not None:
            image_true_2 = image_true_2 - image_true_2.mean(dim=[1, 2, 3], keepdim=True)

            num2 = torch.sum(image_true_2 * image_test, dim=[1, 2, 3])
            den2 = torch.sqrt(
                torch.sum(image_true_2 ** 2, dim=[1, 2, 3]) *
                torch.sum(image_test ** 2, dim=[1, 2, 3])
            ).clamp_min(EPS)

            corr2 = num2 / den2

            value = torch.abs(corr1 - corr2)
        else:
            value = corr1
        self.value = value
        return self.value


class Qabf(BaseMetric):
    """
    Qabf metric (Fusion Quality Index based on gradient magnitude and orientation).

    Computes fusion quality between:
        A (image_true)
        B (image_true_2)
        F (image_test)

    Higher is better.
    """

    higher_is_better: Optional[bool] = True
    is_differentiable = True
    full_state_update = False
    min_arg = 3
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Qabf - Fusion Quality Index"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1

        # Sobel-like kernels
        h1 = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32, device=device)

        h3 = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32, device=device)

        self.register_buffer("h1", h1.view(1, 1, 3, 3))
        self.register_buffer("h3", h3.view(1, 1, 3, 3))

    def _gradient(self, img):
        gx = F.conv2d(img, self.h3, padding=1)
        gy = F.conv2d(img, self.h1, padding=1)

        g = torch.sqrt(gx ** 2 + gy ** 2 + EPS)

        a = torch.atan2(gy, gx + EPS)

        return g, a

    def _qabf_pair(self, aA: Tensor, gA: Tensor, aF: Tensor, gF: Tensor):
        Tg = 0.9994
        kg = -15
        Dg = 0.5

        Ta = 0.9879
        ka = -22
        Da = 0.8

        # Gradient similarity
        GAF = torch.where(gA > gF, gF / (gA + EPS), torch.where(torch.abs(gA - gF) < EPS, gF, gA / (gF + EPS)))

        # Orientation similarity
        AAF = 1 - torch.abs(aA - aF) / (torch.pi / 2)

        # Sigmoid-like mapping
        QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))

        return QgAF * QaAF

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()

        gA, aA = self._gradient(image_true)
        gB, aB = self._gradient(image_true_2)
        gF, aF = self._gradient(image_test)

        QAF = self._qabf_pair(aA, gA, aF, gF)
        QBF = self._qabf_pair(aB, gB, aF, gF)

        numerator = torch.sum(QAF * gA + QBF * gB, dim=[1, 2, 3])
        denominator = torch.sum(gA + gB, dim=[1, 2, 3]) + EPS

        value = numerator / denominator

        self.value = value
        return self.value


class NABF(BaseMetric):
    """
    NABF loss (Noise/Artifact Based Fusion metric)

    Input:
        I1 : (B,1,H,W)
        I2 : (B,1,H,W)
        F  : (B,1,H,W)

    Output:
        Tensor of shape (B,)
    """

    higher_is_better: Optional[bool] = False
    is_differentiable = True
    full_state_update = False
    min_arg = 3
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Nabf - Fusion Quality Index"
        self.commentary = "The lower, the better"
        self.range_min = 0
        self.range_max = 1
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32, device=device)

        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32, device=device)

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.cst = {
            "Td": 2.0,
            "wt_min": 0.001,
            "Lg": 1.5,
            "Nrg": 0.9999,
            "kg": 19,
            "sigmag": 0.5,
            "Nra": 0.9995,
            "ka": 22,
            "sigmaa": 0.5
        }

    def _gradients(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)

        g = torch.sqrt(gx ** 2 + gy ** 2 + EPS)
        a = torch.atan2(gy, gx + EPS)

        return g, a

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        gA, aA = self._gradients(image_true.mean(dim=1, keepdim=True))

        gB, aB = self._gradients(image_true_2.mean(dim=1, keepdim=True))
        gF, aF = self._gradients(image_test.mean(dim=1, keepdim=True))

        # Gradient similarity
        gAF = torch.where((gA == 0) | (gF == 0), torch.zeros_like(gA),
                          torch.where(gA > gF, gF / (gA + EPS), gA / (gF + EPS)))

        gBF = torch.where((gB == 0) | (gF == 0), torch.zeros_like(gB),
                          torch.where(gB > gF, gF / (gB + EPS), gB / (gF + EPS)))

        # Orientation similarity
        aAF = torch.abs(torch.abs(aA - aF) - torch.pi / 2) * 2 / torch.pi
        aBF = torch.abs(torch.abs(aB - aF) - torch.pi / 2) * 2 / torch.pi

        # Quality maps
        QgAF = self.cst['Nrg'] / (1 + torch.exp(-self.cst['kg'] * (gAF - self.cst['sigmag'])))
        QaAF = self.cst['Nra'] / (1 + torch.exp(-self.cst['ka'] * (aAF - self.cst['sigmaa'])))
        QAF = torch.sqrt(QgAF * QaAF + EPS)

        QgBF = self.cst['Nrg'] / (1 + torch.exp(-self.cst['kg'] * (gBF - self.cst['sigmag'])))
        QaBF = self.cst['Nra'] / (1 + torch.exp(-self.cst['ka'] * (aBF - self.cst['sigmaa'])))
        QBF = torch.sqrt(QgBF * QaBF + EPS)

        # Weights
        wtA = torch.where(gA >= self.cst['Td'], gA ** self.cst['Lg'], torch.zeros_like(gA))
        wtB = torch.where(gB >= self.cst['Td'], gB ** self.cst['Lg'], torch.zeros_like(gB))

        wt_sum = torch.sum(wtA + wtB, dim=[1, 2, 3]) + self.cst['wt_min']

        # NABF
        mask = (gF > gA) & (gF > gB)

        self.value = torch.sum(mask * ((1 - QAF) * wtA + (1 - QBF) * wtB), dim=[1, 2, 3]) / wt_sum

        return self.value


class QYang(BaseMetric):
    """
    QYang metric (Fusion Quality Index based on gradient magnitude and orientation).

    Computes fusion quality between:
        A (image_true)
        B (image_true_2)
        F (image_test)

    Higher is better.
    """

    higher_is_better: Optional[bool] = True
    is_differentiable = True
    full_state_update = False
    min_arg = 3
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "QYang - Fusion Quality Index"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1

        # Sobel-like kernels
        h1 = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32)

        h3 = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32)

        self.register_buffer("h1", h1.view(1, 1, 3, 3))
        self.register_buffer("h3", h3.view(1, 1, 3, 3))

    @staticmethod
    def ssim_yang(img1, img2):
        window_size = 7
        sigma = 1.5
        mu1 = gaussian_blur(img1, window_size, sigma)
        mu2 = gaussian_blur(img2, window_size, sigma)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = gaussian_blur(img1 ** 2, window_size, sigma) - mu1_sq
        sigma2_sq = gaussian_blur(img2 ** 2, window_size, sigma) - mu2_sq
        sigma12 = gaussian_blur(img1 * img2, window_size, sigma) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        mssim = ssim_map.mean().item()

        return mssim, ssim_map, sigma1_sq, sigma2_sq

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()

        image_test = image_test.double()
        image_true = image_true.double()
        image_true_2 = image_true_2.double()

        _, ssim_map1, sigma1_sq1, sigma2_sq1 = self.ssim_yang(image_true, image_true_2)
        _, ssim_map2, _, _ = self.ssim_yang(image_true, image_test)
        _, ssim_map3, _, _ = self.ssim_yang(image_true_2, image_test)
        bin_map = (ssim_map1 >= 0.75).double()
        ramda = sigma1_sq1 / (sigma1_sq1 + sigma2_sq1 + 1e-10)

        Q1 = (ramda * ssim_map2 + (1 - ramda) * ssim_map3) * bin_map
        Q2 = torch.max(ssim_map2, ssim_map3) * (1 - bin_map)
        Qy = (Q1 + Q2).mean(dim=[1, 2, 3])
        self.value = Qy
        return self.value


class Qcb(BaseMetric):
    """
    Contrast-based quality metric (Qcb): applies visual sensitivity filter in frequency domain,
    computes local contrast measures, normalizes and compares source vs fused contrasts,
    combines scores with weights and returns mean Qcb.
    """

    higher_is_better: Optional[bool] = True
    is_differentiable = True
    full_state_update = False
    min_arg = 3
    max_arg = 3

    preds: List[Tensor]
    target: List[Tensor]
    target2: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Contrast-based quality metric"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1
        self.cst = {
            'f0': 15.3870,
            'f1': 1.3456,
            'a': 0.7622}

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        # Normalize images to [0, 1]
        image_test = (image_test - image_test.min()) / (image_test.max() - image_test.min())
        image_true = (image_true - image_true.min()) / (image_true.max() - image_true.min())
        image_true_2 = (image_true_2 - image_true_2.min()) / (image_true_2.max() - image_true_2.min())

        h, w = image_test.shape[-2:]

        u, v = torch.meshgrid(torch.fft.fftfreq(h, device=self.device), torch.fft.fftfreq(w, device=self.device),
                              indexing='ij')
        u = u * (h / 30)
        v = v * (w / 30)
        r = torch.sqrt(u ** 2 + v ** 2)
        Sd = (torch.exp(-(r / self.cst['f0']) ** 2) - self.cst['a'] * torch.exp(-(r / self.cst['f1']) ** 2)).to(
            self.device)
        fim1 = torch.fft.ifft2(torch.fft.fft2(image_test) * Sd).real
        fim2 = torch.fft.ifft2(torch.fft.fft2(image_true) * Sd).real
        ffim = torch.fft.ifft2(torch.fft.fft2(image_true_2) * Sd).real
        G1 = self.gaussian2d(2)
        G2 = self.gaussian2d(4)
        C1 = self.contrast(G1, G2, fim1)
        C2 = self.contrast(G1, G2, fim2)
        Cf = self.contrast(G1, G2, ffim)
        C1P = torch.abs(C1 ** 3) / (C1 ** 2 + EPS)
        C2P = torch.abs(C2 ** 3) / (C2 ** 2 + EPS)
        CfP = torch.abs(Cf ** 3) / (Cf ** 2 + EPS)

        mask1 = (C1P < CfP).double()
        Q1F = (C1P / (CfP + EPS)) * mask1 + (CfP / (C1P + EPS)) * (1 - mask1)
        mask2 = (C2P < CfP).double()
        Q2F = (C2P / (CfP + EPS)) * mask2 + (CfP / (C2P + EPS)) * (1 - mask2)
        ramda1 = (C1P ** 2) / (C1P ** 2 + C2P ** 2 + EPS)
        ramda2 = (C2P ** 2) / (C1P ** 2 + C2P ** 2 + EPS)
        Q = ramda1 * Q1F + ramda2 * Q2F
        self.value = Q.mean(dim=[1, 2, 3])
        return self.value

    def gaussian2d(self, sigma):
        """Generates a 2D Gaussian kernel (fixed -15..15 range) used by frequency/contrast computations."""
        x = torch.arange(-15, 16, device=self.device, dtype=torch.float32)
        y = torch.arange(-15, 16, device=self.device, dtype=torch.float32)
        x, y = torch.meshgrid(x, y)
        G = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * torch.pi * sigma ** 2)
        return G[None, None]

    def contrast(self, G1, G2, img):
        """
        Local contrast map: convolve image with two Gaussian kernels G1/G2 and returns contrast measure (buff/buff1 - 1).
        """
        buff = F.conv2d(img, G1, padding=G1.shape[-1] // 2)
        buff1 = F.conv2d(img, G2, padding=G2.shape[-1] // 2)
        buff1 = torch.where(buff1 == 0, EPS, buff1)
        return buff / buff1 - 1


class NIQE(BaseMetric):
    higher_is_better: Optional[bool] = False
    is_differentiable = False
    full_state_update = False
    min_arg = 1
    max_arg = 1

    preds: List[Tensor]

    def __init__(self, device: torch.device):
        super().__init__(device)

        self.metric = "NIQE - Natural Image Quality Evaluator"
        self.commentary = "The lower, the better"
        self.range_min = 0
        self.range_max = 100

        # ---- Load NIQE parameters ----
        self._define_cst(device)
        gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
        self.gam = torch.from_numpy(gam).float().to(device)
        gam_reciprocal = np.reciprocal(gam)
        r_gam = gamma(gam_reciprocal * 2) ** 2 / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))
        self.r_gam = torch.from_numpy(r_gam).float().to(device)

    # ------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------
    def compute(self):
        image_test, _, _ = super().compute()
        x = image_test.to(self.device).float()
        x = self._rgb_to_y(x)

        # scale to [0,255]
        x = (x * 255.).round().clamp(0, 255)

        scores = self._niqe_batch(x)
        self.value = scores
        return scores

    # ------------------------------------------------------------
    def gamma(self, x):
        device = x.device
        x_ = gamma(x.cpu().numpy())
        return torch.from_numpy(x_).float().to(device)

    def _define_cst(self, device):
        niqe_pris_params = np.load(os.path.join(ROOT_DIR, 'Metrics/parameters_niqe.npz'))
        self.pop_mu = torch.from_numpy(np.array([[2.60131368, 0.90570272, 0.81204778, 0.09042671, 0.13873285,
                                                  0.20603212, 0.81896917, 0.06246176, 0.15332711, 0.19590549,
                                                  0.82647245, -0.02552623, 0.18857452, 0.16577625, 0.82429094,
                                                  -0.02536088, 0.18723905, 0.16505118, 2.96949403, 0.96123352,
                                                  0.84935145, 0.08238297, 0.16132155, 0.22492399, 0.85894682,
                                                  0.0550842, 0.17530713, 0.21712828, 0.87207746, -0.03222065,
                                                  0.21548972, 0.18821289, 0.86939884, -0.03232601, 0.21474491,
                                                  0.18677778]])).float().to(device)
        self.pop_cov = torch.from_numpy(np.array([[ 4.53478102e-01,  9.61011041e-02,  8.27632429e-02,  1.53302365e-02,
                                                    3.67237438e-02,  6.32938668e-02,  8.65019827e-02, -4.05766342e-03,
                                                    5.01685177e-02,  5.24100792e-02,  8.68837010e-02, -1.25133889e-02,
                                                    5.81869071e-02,  4.52611186e-02,  8.55508632e-02, -9.79146055e-03,
                                                    5.76796787e-02,  4.42226754e-02,  2.92894703e-01,  6.63504374e-02,
                                                    5.23289337e-02,  3.77972306e-03,  3.33577729e-02,  4.20923209e-02,
                                                    5.24980001e-02, -4.46679594e-03,  3.77941753e-02,  3.80036978e-02,
                                                    5.12099973e-02, -1.07598906e-02,  4.31346464e-02,  3.09489270e-02,
                                                    4.95639566e-02, -6.85393286e-03,  4.13947338e-02,  3.19159692e-02],
                                                  [ 9.61011041e-02,  3.71117101e-02,  2.15529376e-02,  2.40791720e-03,
                                                    1.43592956e-02,  1.93424464e-02,  2.18755973e-02, -1.01965138e-03,
                                                    1.66185030e-02,  1.75088232e-02,  2.13789695e-02, -5.18256634e-03,
                                                    1.91751520e-02,  1.47358040e-02,  2.09638627e-02, -4.92882673e-03,
                                                    1.91779422e-02,  1.43564554e-02,  4.40559610e-02,  2.39119173e-02,
                                                    1.24258734e-02, -5.37124301e-04,  1.14396169e-02,  1.23376610e-02,
                                                    1.21608748e-02, -1.99551051e-03,  1.21907175e-02,  1.15182431e-02,
                                                    1.09794923e-02, -3.72275492e-03,  1.31778493e-02,  9.55312238e-03,
                                                    1.05086233e-02, -3.23382398e-03,  1.28765652e-02,  9.59476188e-03],
                                                  [ 8.27632429e-02,  2.15529376e-02,  1.77071288e-02,  2.99000302e-03,
                                                    8.55846894e-03,  1.37487819e-02,  1.78084622e-02,  5.39359628e-06,
                                                    1.06253025e-02,  1.18018229e-02,  1.71779042e-02, -2.18078128e-03,
                                                    1.22283148e-02,  9.93938669e-03,  1.69117771e-02 -2.04123935e-03,
                                                    1.22431037e-02,  9.61705834e-03,  4.98659710e-02,  1.51256613e-02,
                                                    1.12242797e-02,  3.50981655e-04,  7.78970823e-03,  9.29517174e-03,
                                                    1.11270569e-02, -4.33528630e-04,  8.20537659e-03,  8.74999327e-03,
                                                    1.00308379e-02, -1.86998188e-03,  9.10195601e-03,  7.02151870e-03,
                                                    9.66633329e-03, -1.63727553e-03,  8.99320623e-03,  6.94348741e-03],
                                                  [ 1.53302365e-02,  2.40791720e-03,  2.99000302e-03,  3.94399547e-03,
                                                    -5.03497555e-04,  3.10259311e-03,  4.12479363e-03, -1.53662714e-03,
                                                    2.41384632e-03,  1.09885491e-03,  3.52180035e-03,  5.42976894e-05,
                                                    1.75065910e-03,  1.40801738e-03,  3.55879937e-03, -8.17464852e-05,
                                                    1.81367985e-03  1.30853132e-03  1.07970012e-02  3.06739148e-03,
                                                    2.08089802e-03  2.59141460e-03  2.59284945e-04  2.60992997e-03,
                                                    3.19314290e-03 -1.66556756e-03  2.51427767e-03  1.25729923e-03,
                                                    2.73093547e-03 -4.61237702e-04  1.99401870e-03  1.41934119e-03,
                                                    2.67681128e-03 -4.93088442e-04  2.00001407e-03  1.39500009e-03],
                                                  [ 3.67237438e-02  1.43592956e-02  8.55846894e-03 -5.03497555e-04,
                                                    6.76987335e-03  7.55023504e-03  8.14068921e-03  2.85311895e-04,
                                                    6.52811974e-03  7.49164666e-03  7.80081970e-03 -2.02456851e-03,
                                                    7.81573016e-03  5.99957416e-03  7.60989210e-03 -2.10503069e-03,
                                                    7.85837911e-03  5.80606893e-03  1.34392703e-02  8.77473829e-03,
                                                    4.75093892e-03 -1.37594214e-03  5.05417781e-03  4.39992248e-03,
                                                    4.14640125e-03 -8.08578328e-05  4.33073133e-03  4.67279886e-03,
                                                    3.58193290e-03 -1.17936513e-03  4.84023475e-03  3.68603889e-03,
                                                    3.43130005e-03 -1.19395872e-03  4.82915198e-03  3.63404143e-03],
                                                  [ 6.32938668e-02  1.93424464e-02  1.37487819e-02  3.10259311e-03,
                                                    7.55023504e-03  1.23327840e-02  1.42986789e-02 -1.30640736e-03,
                                                    1.03374648e-02  1.00196045e-02  1.33993580e-02 -2.51257204e-03,
                                                    1.12092449e-02  8.63967926e-03  1.32291670e-02 -2.74184228e-03,
                                                    1.13774690e-02  8.24562908e-03  2.96035185e-02  1.29070136e-02,
                                                    7.97721929e-03  6.49863561e-04  6.28532887e-03  7.75701175e-03,
                                                    8.18849610e-03 -1.63761265e-03  7.52301284e-03  6.72620982e-03,
                                                    7.13677897e-03 -1.91151443e-03  7.70105297e-03  5.68030010e-03,
                                                    6.88394429e-03 -1.89324712e-03  7.66656665e-03  5.60880900e-03],
                                                  [ 8.65019827e-02  2.18755973e-02  1.78084622e-02  4.12479363e-03,
                                                    8.14068921e-03  1.42986789e-02  1.92001016e-02 -1.10079185e-03,
                                                    1.15978879e-02  1.18700244e-02  1.79537439e-02 -2.41281778e-03,
                                                    1.26706128e-02  1.01040091e-02  1.76980961e-02 -2.25497058e-03,
                                                    1.27057419e-02  9.74393442e-03  5.21818687e-02  1.52123425e-02,
                                                    1.12874765e-02  1.49057154e-03  7.22182159e-03  9.67996408e-03,
                                                    1.19133530e-02 -1.52818374e-03  8.91599920e-03  8.53379059e-03,
                                                    1.05266732e-02 -2.06483919e-03  9.32351251e-03  6.99527868e-03,
                                                    1.01702513e-02 -1.64342485e-03  9.10597801e-03  7.04370869e-03],
                                                  [-4.05766342e-03 -1.01965138e-03  5.39359628e-06 -1.53662714e-03,   2.85311895e-04 -1.30640736e-03 -1.10079185e-03  3.83880371e-03,  -2.59772018e-03  6.50189558e-04 -6.06648544e-04  1.36375177e-03,  -1.26553000e-03 -1.47023396e-04 -6.41384156e-04  1.51510135e-03,  -1.44315943e-03  3.21994838e-05  9.50273829e-04  1.66510737e-03,   9.06171187e-04 -1.50438854e-03  1.55027911e-03  2.92957832e-04,  -3.39609540e-05  2.58018145e-03 -6.77811688e-04  1.64607227e-03,   2.19150095e-04  3.84877192e-04  4.56203943e-04  8.58399401e-04,   1.55332263e-04  2.00249409e-04  5.52372402e-04  7.16623537e-04], [ 5.01685177e-02  1.66185030e-02  1.06253025e-02  2.41384632e-03,   6.52811974e-03  1.03374648e-02  1.15978879e-02 -2.59772018e-03,   9.66713617e-03  8.05132019e-03  1.05023416e-02 -2.83370480e-03,   9.83174178e-03  7.08947986e-03  1.03533390e-02 -3.18160357e-03,   1.00716701e-02  6.66162127e-03  2.04698950e-02  9.73565970e-03,   5.67784916e-03  6.11511931e-04  4.56827899e-03  5.78898468e-03,   6.00074097e-03 -2.28455391e-03  6.18525651e-03  4.60610559e-03,   5.15059478e-03 -1.71097580e-03  5.87590391e-03  4.08369369e-03,   4.98945354e-03 -1.63063237e-03  5.79101173e-03  4.11852784e-03], [ 5.24100792e-02  1.75088232e-02  1.18018229e-02  1.09885491e-03,   7.49164666e-03  1.00196045e-02  1.18700244e-02  6.50189558e-04,   8.05132019e-03  9.55746336e-03  1.12833227e-02 -1.91342453e-03,   9.58930178e-03  7.72087507e-03  1.10439469e-02 -1.87710246e-03,   9.57336796e-03  7.51076956e-03  2.55479608e-02  1.21321780e-02,   7.26094654e-03 -6.19835007e-04  6.47941420e-03  6.69392179e-03,   6.91422632e-03 -1.79200983e-04  6.21565931e-03  6.71892593e-03,   6.04556158e-03 -1.51967436e-03  6.90028039e-03  5.34077100e-03,   5.79259094e-03 -1.48763587e-03  6.86776483e-03  5.24892415e-03], [ 8.68837010e-02  2.13789695e-02  1.71779042e-02  3.52180035e-03,   7.80081970e-03  1.33993580e-02  1.79537439e-02 -6.06648544e-04,   1.05023416e-02  1.12833227e-02  1.91177921e-02 -2.77024535e-03,   1.26001229e-02  9.96713928e-03  1.85557526e-02 -2.09607665e-03,   1.23682373e-02  9.71413457e-03  5.63740325e-02  1.47605728e-02,   1.11613778e-02  1.25344171e-03  7.08730148e-03  9.34275707e-03,   1.14447611e-02 -9.37458746e-04  8.26933866e-03  8.36695598e-03,   1.12986897e-02 -2.51362879e-03  9.54617484e-03  6.84014256e-03,   1.08592394e-02 -1.52906024e-03  9.07218925e-03  7.02501584e-03], [-1.25133889e-02 -5.18256634e-03 -2.18078128e-03  5.42976894e-05,  -2.02456851e-03 -2.51257204e-03 -2.41281778e-03  1.36375177e-03,  -2.83370480e-03 -1.91342453e-03 -2.77024535e-03  3.04523582e-03,  -3.66402838e-03 -1.25287810e-03 -2.28579432e-03  6.22343655e-04,  -2.58712555e-03 -1.94915422e-03 -6.13419211e-04 -1.32784261e-03,  -4.67971181e-04  1.88856298e-04 -6.62006438e-04 -6.55690135e-04,  -3.85235480e-04  1.05044241e-03 -1.17114644e-03 -2.13229483e-04,  -6.73618671e-04  1.80361920e-03 -1.52318144e-03  7.26141962e-05,  -3.34407414e-04 -5.49514590e-04 -3.52539659e-04 -8.25379531e-04], [ 5.81869071e-02  1.91751520e-02  1.22283148e-02  1.75065910e-03,   7.81573016e-03  1.12092449e-02  1.26706128e-02 -1.26553000e-03,   9.83174178e-03  9.58930178e-03  1.26001229e-02 -3.66402838e-03,   1.14026330e-02  7.99047770e-03  1.21004801e-02 -2.68663046e-03,   1.09434285e-02  8.01030540e-03  2.49679079e-02  1.18113418e-02,   6.85776633e-03 -5.10846360e-05  5.90911939e-03  6.67623312e-03,   6.72643707e-03 -1.46341817e-03  6.69788611e-03  5.91509910e-03,   6.29960261e-03 -2.49278353e-03  7.30973957e-03  4.81116533e-03,   5.93910237e-03 -1.33242833e-03  6.68465616e-03  5.22159799e-03], [ 4.52611186e-02  1.47358040e-02  9.93938669e-03  1.40801738e-03,   5.99957416e-03  8.63967926e-03  1.01040091e-02 -1.47023396e-04,   7.08947986e-03  7.72087507e-03  9.96713928e-03 -1.25287810e-03,   7.99047770e-03  6.78182560e-03  9.88700584e-03 -2.12437672e-03,   8.44874338e-03  6.22009681e-03  2.34001055e-02  1.01318279e-02,   6.10105255e-03 -7.08245046e-05  5.19314789e-03  5.77395002e-03,   6.03909751e-03 -5.61042152e-04  5.41384274e-03  5.49139244e-03,   5.37450889e-03 -1.04267397e-03  5.75348314e-03  4.64385217e-03,   5.25610943e-03 -1.70994544e-03  6.08484062e-03  4.26824337e-03], [ 8.55508632e-02  2.09638627e-02  1.69117771e-02  3.55879937e-03,   7.60989210e-03  1.32291670e-02  1.76980961e-02 -6.41384156e-04,   1.03533390e-02  1.10439469e-02  1.85557526e-02 -2.28579432e-03,   1.21004801e-02  9.88700584e-03  1.89370422e-02 -2.64772462e-03,   1.26291173e-02  9.53895378e-03  5.55555147e-02  1.44699165e-02,   1.09695508e-02  1.30828385e-03  6.89992799e-03  9.16812458e-03,   1.12940375e-02 -8.50416241e-04  8.02710838e-03  8.22938579e-03,   1.10744408e-02 -1.90108641e-03  9.06021523e-03  6.90085281e-03,   1.10924306e-02 -2.17031334e-03  9.30562974e-03  6.70362647e-03], [-9.79146055e-03 -4.92882673e-03 -2.04123935e-03 -8.17464852e-05,  -2.10503069e-03 -2.74184228e-03 -2.25497058e-03  1.51510135e-03,  -3.18160357e-03 -1.87710246e-03 -2.09607665e-03  6.22343655e-04,  -2.68663046e-03 -2.12437672e-03 -2.64772462e-03  3.48540460e-03,  -4.15188398e-03 -1.11082347e-03 -5.03668404e-04 -1.35017871e-03,  -3.21562467e-04  4.28022603e-04 -7.92251317e-04 -5.37834115e-04,  -2.78995359e-04  8.23098884e-04 -1.01158996e-03 -3.30176237e-04,  -5.09989413e-04 -2.29960148e-04 -5.87799027e-04 -7.33227140e-04,  -9.98303346e-04  1.95217088e-03 -1.69956339e-03 -2.61488929e-05], [ 5.76796787e-02  1.91779422e-02  1.22431037e-02  1.81367985e-03,   7.85837911e-03  1.13774690e-02  1.27057419e-02 -1.44315943e-03,   1.00716701e-02  9.57336796e-03  1.23682373e-02 -2.58712555e-03,   1.09434285e-02  8.44874338e-03  1.26291173e-02 -4.15188398e-03,   1.18820424e-02  7.67056961e-03  2.49263832e-02  1.16632922e-02,   6.73929336e-03 -1.52502867e-04  5.89450165e-03  6.53904450e-03,   6.63093243e-03 -1.36311882e-03  6.52994432e-03  5.89650448e-03,   6.19137993e-03 -1.52006554e-03  6.76324766e-03  5.14622362e-03,   6.26974192e-03 -2.55204646e-03  7.30588945e-03  4.73942287e-03], [ 4.42226754e-02  1.43564554e-02  9.61705834e-03  1.30853132e-03,   5.80606893e-03  8.24562908e-03  9.74393442e-03  3.21994838e-05,   6.66162127e-03  7.51076956e-03  9.71413457e-03 -1.94915422e-03,   8.01030540e-03  6.22009681e-03  9.53895378e-03 -1.11082347e-03,   7.67056961e-03  6.39236839e-03  2.19159754e-02  9.94102179e-03,   5.91775662e-03  8.30827145e-05  4.95737016e-03  5.68102949e-03,   5.85362490e-03 -5.92278115e-04  5.30756971e-03  5.29454791e-03,   5.20426550e-03 -1.62075761e-03  5.87277696e-03  4.24125944e-03,   4.95107040e-03 -9.06604995e-04  5.51007519e-03  4.44785240e-03], [ 2.92894703e-01  4.40559610e-02  4.98659710e-02  1.07970012e-02,   1.34392703e-02  2.96035185e-02  5.21818687e-02  9.50273829e-04,   2.04698950e-02  2.55479608e-02  5.63740325e-02 -6.13419211e-04,   2.49679079e-02  2.34001055e-02  5.55555147e-02 -5.03668404e-04,   2.49263832e-02  2.19159754e-02  8.08880474e-01  7.39340699e-02,   9.13563726e-02  5.50105313e-03  4.09763607e-02  5.44537679e-02,   9.41783648e-02  4.84036459e-03  4.11835042e-02  5.45604350e-02,   1.01080957e-01 -1.71692664e-03  5.24828180e-02  4.77344505e-02,   9.92643727e-02 -5.65902092e-03  5.57356722e-02  4.30180593e-02], [ 6.63504374e-02  2.39119173e-02  1.51256613e-02  3.06739148e-03,   8.77473829e-03  1.29070136e-02  1.52123425e-02  1.66510737e-03,   9.73565970e-03  1.21321780e-02  1.47605728e-02 -1.32784261e-03,   1.18113418e-02  1.01318279e-02  1.44699165e-02 -1.35017871e-03,   1.16632922e-02  9.94102179e-03  7.39340699e-02  2.51503348e-02,   1.50791734e-02  6.24668551e-04  1.14270889e-02  1.34324021e-02,   1.49602434e-02  1.65680389e-04  1.16005783e-02  1.30511101e-02,   1.44658355e-02 -2.21436256e-03  1.34735833e-02  1.10434290e-02,   1.38937878e-02 -2.57194184e-03  1.35580638e-02  1.06080503e-02], [ 5.23289337e-02  1.24258734e-02  1.12242797e-02  2.08089802e-03,   4.75093892e-03  7.97721929e-03  1.12874765e-02  9.06171187e-04,   5.67784916e-03  7.26094654e-03  1.11613778e-02 -4.67971181e-04,   6.85776633e-03  6.10105255e-03  1.09695508e-02 -3.21562467e-04,   6.73929336e-03  5.91775662e-03  9.13563726e-02  1.50791734e-02,   1.60734246e-02  7.12958424e-04  8.40200110e-03  1.06707794e-02,   1.50587832e-02  1.09359732e-03  7.93295843e-03  1.02330200e-02,   1.44014570e-02 -4.07211128e-04  9.34165921e-03  8.44849038e-03,   1.41811104e-02 -8.52147145e-04  9.68004024e-03  7.96665461e-03], [ 3.77972306e-03 -5.37124301e-04  3.50981655e-04  2.59141460e-03,  -1.37594214e-03  6.49863561e-04  1.49057154e-03 -1.50438854e-03,   6.11511931e-04 -6.19835007e-04  1.25344171e-03  1.88856298e-04,  -5.10846360e-05 -7.08245046e-05  1.30828385e-03  4.28022603e-04,  -1.52502867e-04  8.30827145e-05  5.50105313e-03  6.24668551e-04,   7.12958424e-04  3.67606647e-03 -1.63862263e-03  1.54090274e-03,   2.49655142e-03 -1.78231589e-03  1.42159080e-03 -6.30110169e-05,   1.74551549e-03 -4.90711338e-05  5.25006465e-04  3.30747806e-04,   1.86517548e-03  4.14601549e-04  3.12751646e-04  6.24309674e-04], [ 3.33577729e-02  1.14396169e-02  7.78970823e-03  2.59284945e-04,   5.05417781e-03  6.28532887e-03  7.22182159e-03  1.55027911e-03,   4.56827899e-03  6.47941420e-03  7.08730148e-03 -6.62006438e-04,   5.90911939e-03  5.19314789e-03  6.89992799e-03 -7.92251317e-04,   5.89450165e-03  4.95737016e-03  4.09763607e-02  1.14270889e-02,   8.40200110e-03 -1.63862263e-03  7.14344827e-03  6.48058353e-03,   7.18428611e-03  1.15426281e-03  5.32012496e-03  7.16129274e-03,   6.99325529e-03 -8.49552442e-04  6.66936886e-03  5.76466710e-03,   6.63139861e-03 -1.50475496e-03  7.02657263e-03  5.18188368e-03], [ 4.20923209e-02  1.23376610e-02  9.29517174e-03  2.60992997e-03,   4.39992248e-03  7.75701175e-03  9.67996408e-03  2.92957832e-04,   5.78898468e-03  6.69392179e-03  9.34275707e-03 -6.55690135e-04,   6.67623312e-03  5.77395002e-03  9.16812458e-03 -5.37834115e-04,   6.53904450e-03  5.68102949e-03  5.44537679e-02  1.34324021e-02,   1.06707794e-02  1.54090274e-03  6.48058353e-03  8.98343620e-03,   1.07994107e-02 -3.28468385e-04  7.42302255e-03  8.02587311e-03,   9.83652578e-03 -1.01170559e-03  8.08108872e-03  6.79993144e-03,   9.59050655e-03 -1.18571422e-03  8.18951207e-03  6.55631724e-03], [ 5.24980001e-02  1.21608748e-02  1.11270569e-02  3.19314290e-03,   4.14640125e-03  8.18849610e-03  1.19133530e-02 -3.39609540e-05,   6.00074097e-03  6.91422632e-03  1.14447611e-02 -3.85235480e-04,   6.72643707e-03  6.03909751e-03  1.12940375e-02 -2.78995359e-04,   6.63093243e-03  5.85362490e-03  9.41783648e-02  1.49602434e-02,   1.50587832e-02  2.49655142e-03  7.18428611e-03  1.07994107e-02,   1.77877196e-02 -2.92116498e-04  9.10322929e-03  1.04935124e-02,   1.51111441e-02 -2.55216520e-04  9.31523466e-03  8.63061779e-03,   1.48304295e-02 -9.48551527e-04  9.80430502e-03  7.93617159e-03], [-4.46679594e-03 -1.99551051e-03 -4.33528630e-04 -1.66556756e-03,  -8.08578328e-05 -1.63761265e-03 -1.52818374e-03  2.58018145e-03,  -2.28455391e-03 -1.79200983e-04 -9.37458746e-04  1.05044241e-03,  -1.46341817e-03 -5.61042152e-04 -8.50416241e-04  8.23098884e-04,  -1.36311882e-03 -5.92278115e-04  4.84036459e-03  1.65680389e-04,   1.09359732e-03 -1.78231589e-03  1.15426281e-03 -3.28468385e-04,  -2.92116498e-04  3.24341389e-03 -1.64432303e-03  1.22453089e-03,   5.55442959e-04  8.48128042e-04 -2.34601301e-04  5.71447933e-04,   4.58167234e-04  3.28092453e-04  4.34340358e-05  2.14649046e-04], [ 3.77941753e-02  1.21907175e-02  8.20537659e-03  2.51427767e-03,   4.33073133e-03  7.52301284e-03  8.91599920e-03 -6.77811688e-04,   6.18525651e-03  6.21565931e-03  8.26933866e-03 -1.17114644e-03,   6.69788611e-03  5.41384274e-03  8.02710838e-03 -1.01158996e-03,   6.52994432e-03  5.30756971e-03  4.11835042e-02  1.16005783e-02,   7.93295843e-03  1.42159080e-03  5.32012496e-03  7.42302255e-03,   9.10322929e-03 -1.64432303e-03  7.17284710e-03  6.46211525e-03,   7.61464748e-03 -1.33118320e-03  7.03583016e-03  5.53532008e-03,   7.38373814e-03 -1.37680984e-03  7.04062836e-03  5.40831632e-03], [ 3.80036978e-02  1.15182431e-02  8.74999327e-03  1.25729923e-03,   4.67279886e-03  6.72620982e-03  8.53379059e-03  1.64607227e-03,   4.60610559e-03  6.71892593e-03  8.36695598e-03 -2.13229483e-04,   5.91509910e-03  5.49139244e-03  8.22938579e-03 -3.30176237e-04,   5.89650448e-03  5.29454791e-03  5.45604350e-02  1.30511101e-02,   1.02330200e-02 -6.30110169e-05  7.16129274e-03  8.02587311e-03,   1.04935124e-02  1.22453089e-03  6.46211525e-03  8.65945069e-03,   9.44820009e-03 -5.14490992e-04  7.61330052e-03  6.94209913e-03,   9.04956669e-03 -1.26334996e-03  8.03896583e-03  6.25417170e-03], [ 5.12099973e-02  1.09794923e-02  1.00308379e-02  2.73093547e-03,   3.58193290e-03  7.13677897e-03  1.05266732e-02  2.19150095e-04,   5.15059478e-03  6.04556158e-03  1.12986897e-02 -6.73618671e-04,   6.29960261e-03  5.37450889e-03  1.10744408e-02 -5.09989413e-04,   6.19137993e-03  5.20426550e-03  1.01080957e-01  1.44658355e-02,   1.44014570e-02  1.74551549e-03  6.99325529e-03  9.83652578e-03,   1.51111441e-02  5.55442959e-04  7.61464748e-03  9.44820009e-03,   1.73792581e-02 -1.01801631e-03  1.00024318e-02  8.51219915e-03,   1.61124363e-02 -1.15705337e-03  9.85743084e-03  7.75885746e-03], [-1.07598906e-02 -3.72275492e-03 -1.86998188e-03 -4.61237702e-04,  -1.17936513e-03 -1.91151443e-03 -2.06483919e-03  3.84877192e-04,  -1.71097580e-03 -1.51967436e-03 -2.51362879e-03  1.80361920e-03,  -2.49278353e-03 -1.04267397e-03 -1.90108641e-03 -2.29960148e-04,  -1.52006554e-03 -1.62075761e-03 -1.71692664e-03 -2.21436256e-03,  -4.07211128e-04 -4.90711338e-05 -8.49552442e-04 -1.01170559e-03,  -2.55216520e-04  8.48128042e-04 -1.33118320e-03 -5.14490992e-04,  -1.01801631e-03  2.46738473e-03 -2.24311189e-03 -7.53966271e-05,  -4.04770761e-04 -5.72412262e-04 -7.01174936e-04 -1.14884961e-03], [ 4.31346464e-02  1.31778493e-02  9.10195601e-03  1.99401870e-03,   4.84023475e-03  7.70105297e-03  9.32351251e-03  4.56203943e-04,   5.87590391e-03  6.90028039e-03  9.54617484e-03 -1.52318144e-03,   7.30973957e-03  5.75348314e-03  9.06021523e-03 -5.87799027e-04,   6.76324766e-03  5.87277696e-03  5.24828180e-02  1.34735833e-02,   9.34165921e-03  5.25006465e-04  6.66936886e-03  8.08108872e-03,   9.31523466e-03 -2.34601301e-04  7.03583016e-03  7.61330052e-03,   1.00024318e-02 -2.24311189e-03  8.76166931e-03  6.39457453e-03,   9.06889308e-03 -1.13568633e-03  8.04097578e-03  6.45294467e-03], [ 3.09489270e-02  9.55312238e-03  7.02151870e-03  1.41934119e-03,   3.68603889e-03  5.68030010e-03  6.99527868e-03  8.58399401e-04,   4.08369369e-03  5.34077100e-03  6.84014256e-03  7.26141962e-05,   4.81116533e-03  4.64385217e-03  6.90085281e-03 -7.33227140e-04,   5.14622362e-03  4.24125944e-03  4.77344505e-02  1.10434290e-02,   8.44849038e-03  3.30747806e-04  5.76466710e-03  6.79993144e-03,   8.63061779e-03  5.71447933e-04  5.53532008e-03  6.94209913e-03,   8.51219915e-03 -7.53966271e-05  6.39457453e-03  6.13540817e-03,   8.08421419e-03 -1.59744992e-03  7.10914766e-03  5.12813603e-03], [ 4.95639566e-02  1.05086233e-02  9.66633329e-03  2.67681128e-03,   3.43130005e-03  6.88394429e-03  1.01702513e-02  1.55332263e-04,   4.98945354e-03  5.79259094e-03  1.08592394e-02 -3.34407414e-04,   5.93910237e-03  5.25610943e-03  1.10924306e-02 -9.98303346e-04,   6.26974192e-03  4.95107040e-03  9.92643727e-02  1.38937878e-02,   1.41811104e-02  1.86517548e-03  6.63139861e-03  9.59050655e-03,   1.48304295e-02  4.58167234e-04  7.38373814e-03  9.04956669e-03,   1.61124363e-02 -4.04770761e-04  9.06889308e-03  8.08421419e-03,   1.72091342e-02 -1.70246944e-03  1.02114984e-02  7.69302597e-03], [-6.85393286e-03 -3.23382398e-03 -1.63727553e-03 -4.93088442e-04,  -1.19395872e-03 -1.89324712e-03 -1.64342485e-03  2.00249409e-04,  -1.63063237e-03 -1.48763587e-03 -1.52906024e-03 -5.49514590e-04,  -1.33242833e-03 -1.70994544e-03 -2.17031334e-03  1.95217088e-03,  -2.55204646e-03 -9.06604995e-04 -5.65902092e-03 -2.57194184e-03,  -8.52147145e-04  4.14601549e-04 -1.50475496e-03 -1.18571422e-03,  -9.48551527e-04  3.28092453e-04 -1.37680984e-03 -1.26334996e-03,  -1.15705337e-03 -5.72412262e-04 -1.13568633e-03 -1.59744992e-03,  -1.70246944e-03  2.78698695e-03 -2.83206713e-03 -3.55635451e-04], [ 4.13947338e-02  1.28765652e-02  8.99320623e-03  2.00001407e-03,   4.82915198e-03  7.66656665e-03  9.10597801e-03  5.52372402e-04,   5.79101173e-03  6.86776483e-03  9.07218925e-03 -3.52539659e-04,   6.68465616e-03  6.08484062e-03  9.30562974e-03 -1.69956339e-03,   7.30588945e-03  5.51007519e-03  5.57356722e-02  1.35580638e-02,   9.68004024e-03  3.12751646e-04  7.02657263e-03  8.18951207e-03,   9.80430502e-03  4.34340358e-05  7.04062836e-03  8.03896583e-03,   9.85743084e-03 -7.01174936e-04  8.04097578e-03  7.10914766e-03,   1.02114984e-02 -2.83206713e-03  9.33443877e-03  6.14234154e-03], [ 3.19159692e-02  9.59476188e-03  6.94348741e-03  1.39500009e-03,   3.63404143e-03  5.60880900e-03  7.04370869e-03  7.16623537e-04,   4.11852784e-03  5.24892415e-03  7.02501584e-03 -8.25379531e-04,   5.22159799e-03  4.26824337e-03  6.70362647e-03 -2.61488929e-05,   4.73942287e-03  4.44785240e-03  4.30180593e-02  1.06080503e-02,   7.96665461e-03  6.24309674e-04  5.18188368e-03  6.55631724e-03,   7.93617159e-03  2.14649046e-04  5.40831632e-03  6.25417170e-03,   7.75885746e-03 -1.14884961e-03  6.45294467e-03  5.12813603e-03,   7.69302597e-03 -3.55635451e-04  6.14234154e-03  5.42249259e-03]])).float().to(device)
        self.kernel = torch.from_numpy(niqe_pris_params['gaussian_window']).float().to(device)

    def _niqe_batch(self, x):
        B, _, H, W = x.shape

        block = 96
        num_h = H // block
        num_w = W // block
        x = x[:, :, :num_h * block, :num_w * block]

        distparam = []

        for scale in [1, 2]:

            mscn = self._compute_mscn(x)

            patches = F.unfold(
                mscn,
                kernel_size=block // scale,
                stride=block // scale
            ).transpose(1, 2).view(B, num_h * num_w, block // scale, block // scale)  # (B, N, D)

            feats = self._compute_features(patches)
            distparam.extend(feats)

            if scale == 1:
                x = F.interpolate(
                    x / 255.0,
                    scale_factor=0.5,
                    mode="bicubic",
                    align_corners=False,
                    antialias=True
                ) * 255.0

        distparam = torch.stack(distparam, dim=-1)  # (B, N, F)

        mu_dist = torch.nanmean(distparam, dim=1)

        cov_dist = torch.stack([
            torch.cov(d[~torch.isnan(d).any(dim=1)].T)
            for d in distparam
        ])

        cov_avg = (self.pop_cov + cov_dist) / 2
        invcov = torch.linalg.pinv(cov_avg)

        diff = self.pop_mu - mu_dist

        scores = torch.sqrt(
            torch.einsum("bi,bij,bj->b", diff, invcov, diff)
        )

        return scores

    # ------------------------------------------------------------
    def _compute_features(self, patches):
        # patches: (B, N, Ph, Pw)
        feat = []
        alpha, beta_l, beta_r = self.estimate_aggd_param(patches)
        feat.extend([alpha, (beta_l + beta_r) / 2])

        # distortions disturb the fairly regular structure of natural images.
        # This deviation can be captured by analyzing the sample distribution of
        # the products of pairs of adjacent coefficients computed along
        # horizontal, vertical and diagonal orientations.
        shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
        for shift in shifts:
            shifted_patches = torch.roll(patches, shift, dims=(-2, -1))
            alpha, beta_l, beta_r = self.estimate_aggd_param(patches * shifted_patches)
            # Eq. 8
            mean = (beta_r - beta_l) * (self.gamma(2 / alpha) / self.gamma(1 / alpha))
            feat.extend([alpha, mean, beta_l, beta_r])
        return feat

    def estimate_aggd_param(self, patches):
        """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.

            Args:
                block (ndarray): 2D Image block.

            Returns:
                tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
                    distribution (Estimating the parames in Equation 7 in the paper).
            """
        patches = patches.flatten(2)  # (B, N, Ph*Pw)
        B, N, P = patches.shape

        neg_patches = patches < 0
        pos_patches = patches > 0
        left_std = torch.sqrt(torch.sum((patches * neg_patches) ** 2, dim=-1) / torch.sum(neg_patches, dim=-1))
        right_std = torch.sqrt(torch.sum((patches * pos_patches) ** 2, dim=-1) / torch.sum(pos_patches, dim=-1))
        gammahat = left_std / right_std
        rhat = ((torch.sum(torch.abs(patches), dim=-1) / patches.shape[-1]) ** 2 /
                (torch.sum(patches ** 2, dim=-1) / patches.shape[-1]))
        rhatnorm = (rhat * (gammahat ** 3 + 1) * (gammahat + 1)) / ((gammahat ** 2 + 1) ** 2)
        array_position = torch.argmin((self.r_gam[None, None].repeat(B, N, 1) - rhatnorm[..., None]) ** 2, dim=-1)

        alphas = self.gam[array_position]
        beta_l = left_std * torch.sqrt(self.gamma(1 / alphas) / self.gamma(3 / alphas))
        beta_r = right_std * torch.sqrt(self.gamma(1 / alphas) / self.gamma(3 / alphas))
        return alphas, beta_l, beta_r  # [B, 3]

    def _compute_mscn(self, x):
        B, C, H, W = x.shape
        k = self.kernel.shape[-1]
        kernel = self.kernel.view(1, 1, k, k).repeat(C, 1, 1, 1)

        mu = F.conv2d(x, kernel, padding=k // 2, groups=C)
        sigma = F.conv2d(x * x, kernel, padding=k // 2, groups=C)
        sigma = torch.sqrt(torch.abs(sigma - mu ** 2))

        return (x - mu) / (sigma + 1)

    def _rgb_to_y(self, x):
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
