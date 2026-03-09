import gc
from typing import Optional, Union, Sequence, List, Any
from warnings import warn

import torch
import torchvision
from kornia.filters import joint_bilateral_blur
from torch import Tensor, softmax, nn, tensor
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy, conv2d
from torch_similarity.modules import GradientCorrelationLoss2d
from torchmetrics import MeanSquaredError, Metric
from torchmetrics.clustering import MutualInfoScore, NormalizedMutualInfoScore
from torchmetrics.functional.image.scc import _scc_update, _scc_per_channel_compute
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchvision import models
from torchvision.transforms.v2.functional import gaussian_blur

from ..Image import ImageTensor
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
        self.ratio_list = torch.tensor([1, 3 / 4, 2 / 3, 9 / 16, 9 / 21])
        self.ratio_dict = {1: [512, 512],
                           round(3 / 4, 3): [480, 640],
                           round(2 / 3, 3): [448, 672],
                           round(9 / 16, 3): [405, 720],
                           round(9 / 21, 3): [340, 800]}
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
                print(f"Warning: More than 1 argument was given to the metric {self.metric}, "
                      f"only the first will be used as input")
                preds = args[0]
                target2, target = None, None
            elif self.max_arg == 2:
                print(
                    f"Warning: More than 2 arguments were given to the metric {self.metric}, only the first two will be used as target and preds")
                target, preds = args[:2]
                target2 = None
            elif len(args) > self.max_arg:
                print(
                    f"Warning: More than 3 arguments were given to the metric {self.metric}, only the first two will be used as target and preds")
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

        size = self._determine_size_from_ratio(image_test)
        image_test = image_test.resize(size).to_tensor()
        image_true = image_true.resize(size).to_tensor() if image_true is not None else None
        image_true_2 = image_true_2.resize(size).to_tensor() if image_true_2 is not None else None
        if mask is not None:
            mask = ImageTensor(mask * 1.)
            self.mask = mask.resize(size).to_tensor().to(torch.bool).to(self.device)
        else:
            self.mask = torch.ones_like(image_test, device=self.device).to(torch.bool)
        if weights is not None:
            weights = ImageTensor(weights / weights.max())
            self.weights = weights.resize(size).to_tensor().to(self.device)
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
            value_2 = torch.stack([self.mi(img_true2, img_test) for img_true2, img_test in zip(image_true_2, image_test)], dim=0)
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
            value_2 = torch.stack([self.nmi(img_true2, img_test) for img_true2, img_test in zip(image_true_2, image_test)], dim=0)
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
        ref_true = grad_tensor(ImageTensor(img_true, batched=img_true.shape[0] > 1, device=self.device)) * self.mask[:, :2]
        ref_test = grad_tensor(ImageTensor(img_test, batched=img_test.shape[0] > 1, device=self.device)) * self.mask[:, :2]
        dot_prod = torch.abs(torch.cos(ref_true[:, 1] - ref_test[:, 1])) * 2 - 1
        image_nec = ref_true[:, 0] * ref_test[:, 0] * dot_prod * self.weights[:, 0] * self.mask[:, 0]
        nec_ref = torch.sqrt(torch.abs(torch.sum(ref_true[:, 0] ** 2 * self.weights[:, 0] * self.mask[:, 0], dim=[-1, -2]) *
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
            high_pass_filter = tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

        self.hp_filter = high_pass_filter
        self.ws = window_size

        self.add_state("scc_score", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, *args,
               mask=None, weights=None, **kwargs) -> None:
        super().update(*args, mask=mask, weights=weights, **kwargs)

    def _compute_scc(self, preds, target):
        preds, target, hp_filter = _scc_update(preds, target, self.hp_filter, self.ws)
        scc_per_channel = [
            _scc_per_channel_compute(preds[:, i, :, :].unsqueeze(1), target[:, i, :, :].unsqueeze(1), hp_filter, self.ws)
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
        self.metric = "Binary cross-entropy"
        self.commentary = "The higher, the better"
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
        denominator = torch.sqrt(torch.sum((img1 - mean1) ** 2, dim=[1, 2, 3]) * torch.sum((img2 - mean2) ** 2, dim=[1, 2, 3]))
        return numerator / (denominator + EPS)

    def scale(self):
        self.range_max += self.range_max
        return self


class ShannonEntropy(BaseMetric):
    """Tsallis (or Shannon when q=1) entropies computed per image from histogram of size `ksize`."""
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
        self.metric = "Binary cross-entropy"
        self.commentary = "The higher, the better"
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

    def update(self, *args, mask=None, **kwargs) -> None:
        super().update(*args, mask=mask, **kwargs)

    def compute(self):
        image_test, image_true, image_true_2 = super().compute()
        value = self.vifp_mscale(image_true * self.mask * self.weights,
                          image_test * self.mask * self.weights)
        if image_true_2 is not None:
            value_2 = self.vifp_mscale(image_true_2 * self.mask * self.weights,
                                image_test * self.mask * self.weights)
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


class StrcturalCorrelationDifference(BaseMetric):
    """Structural correlation difference: computes Pearson correlation between A–F and B–F and returns their absolute difference."""
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
        self.metric = "Structural Correlation Difference - Absolute difference of Pearson correlations"
        self.commentary = "The higher, the better"
        self.range_min = -2
        self.range_max = 2

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
            value_2 = torch.abs(torch.sum(image_true_2 * image_test, dim=[1, 2, 3]) / torch.sqrt(
                torch.sum(image_true_2 ** 2, dim=[1, 2, 3]) *
                torch.sum(image_test ** 2, dim=[1, 2, 3])) + EPS)
            value = torch.abs(value - value_2)
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
        self.metric = "Nabf - Fusion Quality Index"
        self.commentary = "The higher, the better"
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
        Sd = (torch.exp(-(r / self.cst['f0']) ** 2) - self.cst['a'] * torch.exp(-(r / self.cst['f1']) ** 2)).to(self.device)
        fim1 = torch.fft.ifft2(torch.fft.fft2(image_test) * Sd).real
        fim2 = torch.fft.ifft2(torch.fft.fft2(image_true) * Sd).real
        ffim = torch.fft.ifft2(torch.fft.fft2(image_true_2) * Sd).real
        G1 = self.gaussian2d(2)
        G2 = self.gaussian2d(4)
        C1 = self.contrast(G1, G2, fim1)
        C2 = self.contrast(G1, G2, fim2)
        Cf = self.contrast(G1, G2, ffim)
        C1P = (torch.abs(C1) ** 3) / (h * C1 ** 2 + EPS)
        C2P = (torch.abs(C2) ** 3) / (h * C2 ** 2 + EPS)
        CfP = (torch.abs(Cf) ** 3) / (h * Cf ** 2 + EPS)

        mask1 = (C1P < CfP).double()
        Q1F = (C1P / CfP) * mask1 + (CfP / C1P) * (1 - mask1)
        mask2 = (C2P < CfP).double()
        Q2F = (C2P / CfP) * mask2 + (CfP / C2P) * (1 - mask2)
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
        buff1 = torch.where(buff1 == 0, torch.ones_like(buff1) * EPS, buff1)
        return buff / buff1 - 1
