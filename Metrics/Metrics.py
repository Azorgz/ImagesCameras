import gc
from typing import Optional, Union, Sequence, List

import torch
import torchvision
from kornia.filters import joint_bilateral_blur
from torch import Tensor, softmax, nn
from torch.masked import masked_tensor
from torch_similarity.modules import GradientCorrelationLoss2d
from torchmetrics import MeanSquaredError, Metric
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.image import SpatialCorrelationCoefficient

from ..Image import ImageTensor
######################### METRIC ##############################################
from ..tools.gradient_tools import grad_tensor


class BaseMetric(Metric):
    ##
    # A class defining the general basic metric working with Tensor on GPU
    # Set to True if the metric is differentiable else set to False
    is_differentiable: Optional[bool] = None

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, device: torch.device = None, **kwargs):
        super().__init__()
        self.target = None
        self.preds = None
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
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

    def update(self, target: ImageTensor, preds: ImageTensor, *args, mask=None, weights=None, **kwargs) -> None:
        target = ImageTensor(target)
        preds = ImageTensor(preds)
        if preds.channel_num == target.channel_num:
            image_true = target
            image_test = preds
        elif target.channel_num > 1:
            image_true = ImageTensor(target.mean(dim=target.channel_pos))
            image_test = preds
        else:
            image_true = target
            image_test = ImageTensor(preds.mean(dim=preds.channel_pos))

        size = self._determine_size_from_ratio(image_true)
        image_true = image_true.resize(size).to_tensor()
        image_test = image_test.resize(size).to_tensor()
        if mask is not None:
            mask = ImageTensor(mask * 1.)
            self.mask = mask.resize(size).to_tensor().to(torch.bool).to(self.device)

        else:
            self.mask = torch.ones_like(image_true, device=self.device).to(torch.bool)

        if weights is not None:
            weights = ImageTensor(weights / weights.max())
            self.weights = weights.resize(size).to_tensor().to(self.device)
        else:
            self.weights = torch.ones_like(image_true, device=self.device)

        self.preds.append(image_test)
        self.target.append(image_true)

    def compute(self):
        im1 = self.preds[-1]
        im2 = self.target[-1]
        self.preds = []
        self.target = []
        return im1, im2

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

    preds: List[Tensor]
    target: List[Tensor]

    return_image: Optional[bool] = True

    def __init__(self, device: torch.device, abs_values: bool = False, no_negative_values: bool = False):
        super().__init__(device)
        self.ssim = StructuralSimilarityIndexMeasure(gaussian_kernel=True,
                                                     sigma=1.5,
                                                     kernel_size=11,
                                                     reduction=None,
                                                     data_range=None,
                                                     k1=0.01, k2=0.03,
                                                     return_full_image=True,
                                                     return_contrast_sensitivity=False).to(self.device)
        self.metric = "SSIM"
        self.commentary = "The higher, the better"
        self.abs_values = abs_values
        self.no_negative_values = no_negative_values

    def update(self, preds: ImageTensor, target: ImageTensor, *args, mask=None, return_image=False, **kwargs) -> None:
        super().update(preds, target, *args, mask=mask, **kwargs)
        self.return_image = return_image

    def compute(self):
        image_test, image_true = super().compute()
        _, image = self.ssim(image_test, image_true)
        image = torch.abs(image)
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

    preds: List[Tensor]
    target: List[Tensor]

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

    def update(self, preds: ImageTensor, target: ImageTensor, *args, mask=None, **kwargs) -> None:
        super().update(preds, target, *args, mask=mask, **kwargs)

    def compute(self):
        image_test, image_true = super().compute()
        self.value = self.ms_ssim(image_test * self.mask * self.weights,
                                  image_true * self.mask * self.weights)
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

    return_image: Optional[bool] = True

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "MSE"
        self.range_max = 1
        self.commentary = "The lower, the better"

    def update(self, preds: ImageTensor, target: ImageTensor, *args, mask=None, return_image=False, **kwargs) -> None:
        super().update(preds, target, *args, mask=mask, **kwargs)
        self.return_image = return_image

    def compute(self):
        image_test, image_true = super().compute()
        diff = (image_test - image_true) * (self.mask * 1.)
        image_mse = diff ** 2
        self.value = torch.mean(image_mse)
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

    return_image: Optional[bool] = True

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.rmse = MeanSquaredError(squared=False).to(self.device)
        self.metric = "RMSE"
        self.range_max = 1

    def update(self, preds: ImageTensor, target: ImageTensor, *args, mask=None, return_image=False, **kwargs) -> None:
        super().update(preds, target, *args, mask=mask, **kwargs)
        self.return_image = return_image

    def compute(self):
        image_test, image_true = super().compute()
        diff = (image_test - image_true) * (self.mask * 1.)
        image_mse = torch.abs(diff)
        self.value = torch.sqrt(torch.mean(image_mse ** 2, dim=(1, 2, 3)))
        self.reset()
        if self.return_image:
            return ImageTensor(image_mse.mean(dim=1, keepdim=True)).RGB('gray')
        return self.value

    def scale(self):
        self.value = super().scale
        return self.value


class PSNR(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.psnr = PeakSignalNoiseRatio(data_range=None, base=10.0, reduction=None, dim=None).to(self.device)
        self.metric = "Peak Signal Noise Ratio"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = "inf"

    def update(self, preds: ImageTensor, target: ImageTensor, *args, mask=None, **kwargs) -> None:
        super().update(preds, target, *args, mask=mask, **kwargs)

    def compute(self):
        image_test, image_true = super().compute()
        try:
            image_true = image_true * self.mask
            image_test = image_test * self.mask
            self.value = self.psnr(image_true, image_test)
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

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Edges Correlation"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1
        self.return_image = False
        self.return_coeff = False

    def update(self, preds: ImageTensor, target: ImageTensor, *args,
               mask=None, weights=None, return_image=False, return_coeff=False, **kwargs) -> None:
        super().update(preds, target, *args, mask=mask, weights=weights, **kwargs)
        self.return_image = return_image
        self.return_coeff = return_coeff

    def compute(self):
        image_test, image_true = super().compute()
        try:
            image_test = joint_bilateral_blur(image_test, image_true, (3, 3), 0.1, (1.5, 1.5))
            image_true = joint_bilateral_blur(image_true, image_test, (3, 3), 0.1, (1.5, 1.5))
        except torch.OutOfMemoryError:
            pass
        ref_true = grad_tensor(
            ImageTensor(image_true, batched=image_true.shape[0] > 1, device=self.device)) * self.mask[:, :2]
        ref_test = grad_tensor(
            ImageTensor(image_test, batched=image_test.shape[0] > 1, device=self.device)) * self.mask[:, :2]
        weights = self.weights[:, 0] * self.mask[:, 0]
        dot_prod = torch.abs(torch.cos(ref_true[:, 1] - ref_test[:, 1]))
        image_nec = ref_true[:, 0] * ref_test[:, 0] * dot_prod * weights
        nec_ref = torch.sqrt(torch.abs(torch.sum(ref_true[:, 0] * ref_true[:, 0] * weights, dim=[-1, -2]) *
                                       torch.sum(ref_test[:, 0] * ref_test[:, 0] * weights, dim=[-1, -2])) + 1e-6)
        self.value = (image_nec.sum(dim=[-1, -2]) / nec_ref)
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

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Spatial Correlation Coefficient"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1
        self.return_image = False
        self.return_coeff = False
        self.scc = SpatialCorrelationCoefficient().to(device)

    def update(self, preds: ImageTensor, target: ImageTensor, *args,
               mask=None, weights=None, return_image=False, return_coeff=False, **kwargs) -> None:
        super().update(preds, target, *args, mask=mask, weights=weights, **kwargs)
        self.return_image = return_image
        self.return_coeff = return_coeff

    def compute(self):
        image_test, image_true = super().compute()
        self.value = self.scc(image_test * self.mask * self.weights, image_true * self.mask * self.weights)
        return self.value

    def scale(self):
        self.range_max += self.range_max
        return self


class GradientCorrelation(BaseMetric, GradientCorrelationLoss2d):

    def __init__(self, device: torch.device):
        super(GradientCorrelation, self).__init__(return_map=True, device=device)
        self.return_image = True

    def forward(self, x, y, mask=None, weights=None):
        _, gc_map = super().forward(x, y)

        if weights is not None:
            gc_map_ = gc_map * weights
        else:
            gc_map_ = gc_map
        if mask is not None:
            gc_map_ *= mask
        return gc_map_

    def update(self, preds: ImageTensor, target: ImageTensor, *args,
               mask=None, weights=None, return_image=False, **kwargs) -> None:
        super().update(preds, target, *args, mask=mask, weights=weights, **kwargs)
        self.return_image = return_image

    def compute(self):
        image_test, image_true = super().compute()
        res = self.forward(image_test, image_true,
                           mask=self.mask,
                           weights=self.weights)
        self.value = torch.abs(res.flatten(1, -1).sum(-1))
        if self.return_image:
            return res
        else:
            return self.value


class VGGLoss(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = False
    is_differentiable = True
    full_state_update = False

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.metric = "Semantic loss"
        self.commentary = "The lower, the better"
        self.range_min = 0
        self.range_max = 1
        self.vgg = torchvision.models.vgg19(weights='IMAGENET1K_V1').to(device)
        self.rmse = MeanSquaredError(squared=False).to(device)
        self.max = nn.Softmax()

    def update(self, preds: ImageTensor, target: ImageTensor, *args, **kwargs) -> None:
        super().update(preds, target, *args, **kwargs)

    def compute(self):
        image_test, image_true = super().compute()
        ref_sem = self.max(self.vgg(image_true))
        test_sem = self.max(self.vgg(image_test))
        self.value = self.rmse(ref_sem, test_sem)
        return self.value

    def scale(self):
        self.range_max += self.range_max
        return self