from typing import Optional, Union, Sequence

import torch
from torch import Tensor
from torchmetrics import MeanSquaredError as MSE, Metric
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

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

    def __init__(self, device=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.metric = "Base Metric"
        self.mask = None
        self.value = 0
        self.range_min = 0
        self.range_max = 1
        self.commentary = "Just a base"
        self.ratio_list = torch.tensor([1, 3 / 4, 2 / 3, 9 / 16, 9 / 21])
        self.ratio_dict = {1: [512, 512],
                           round(3 / 4, 3): [480, 640],
                           round(2 / 3, 3): [440, 660],
                           round(9 / 16, 3): [405, 720],
                           round(9 / 21, 3): [340, 800]}
        self.to(device)

    def update(self, preds: ImageTensor, target: ImageTensor, *args, mask=None, **kwargs) -> None:
        if preds.channel_num == target.channel_num:
            image_true = preds
            image_test = target
        elif preds.channel_num > 1:
            image_true = preds.GRAY()
            image_test = target
        else:
            image_true = preds
            image_test = target.GRAY()

        size = self._determine_size_from_ratio(image_true)
        image_true = image_true.resize(size).to_tensor()
        image_test = image_test.resize(size).to_tensor()
        if mask is not None:
            mask = mask * 1.
            self.mask = mask.resize(size).to_tensor().to(torch.bool)

        self.preds.append(image_true)
        self.target.append(image_test)

    def compute(self):
        im1 = self.preds[-1]
        im2 = self.target[-1]
        self.preds = []
        self.target = []
        return im1, im2

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
             ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        return self._plot(val, ax)

    # def __call__(self, im1, im2, *args, mask=None, **kwargs):
    #     # Input array is a path to an image OR an already formed ndarray instance
    #     # assert im1.shape[-2:] == im2.shape[-2:], " The inputs are not the same size"
    #
    #     if im1.channel_num == im2.channel_num:
    #         self.image_true = im1
    #         self.image_test = im2
    #     elif im1.channel_num > 1:
    #         self.image_true = im1.GRAY()
    #         self.image_test = im2
    #     else:
    #         self.image_true = im1
    #         self.image_test = im2.GRAY()
    #
    #     size = self._determine_size_from_ratio()
    #     self.image_true = self.image_true.resize(size).to_tensor()
    #     self.image_test = self.image_test.resize(size).to_tensor()
    #     if mask is not None:
    #         mask = mask * 1.
    #         self.mask = mask.resize(size).to_tensor().to(torch.bool)
    #     self.value = 0

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


class Metric_ssim_tensor(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    return_image: Optional[bool] = True

    def __init__(self, device):
        super().__init__(device)
        self.ssim = SSIM(gaussian_kernel=True,
                         sigma=1.5,
                         kernel_size=11,
                         reduction=None,
                         data_range=None,
                         k1=0.01, k2=0.03,
                         return_full_image=True,
                         return_contrast_sensitivity=False).to(self.device)
        self.metric = "SSIM"
        self.commentary = "The higher, the better"

    # def __call__(self, im1, im2, *args, mask=None, return_image=False, **kwargs):
    #     super().__call__(im1, im2, *args, mask=mask, **kwargs)
    #     if mask is None:
    #         temp, image = self.ssim(self.image_test, self.image_true)
    #         self.value = torch.abs(image).mean()
    #         self.ssim.reset()
    #     else:
    #         temp, image = self.ssim(self.image_test, self.image_true)
    #         image = torch.abs(image)
    #         self.value = image[:, :, self.mask[0, 0, :, :]].mean()
    #         self.ssim.reset()
    #     del temp
    #     # self.value = self.ssim(self.image_test * mask, self.image_true * mask)
    #     if return_image:
    #         return ImageTensor(image, permute_image=True).RGB('gray')
    #     else:
    #         return self.value

    def update(self, preds: ImageTensor, target: ImageTensor, *args, mask=None, return_image=False, **kwargs) -> None:
        super().update(preds, target, *args, mask=mask, **kwargs)
        self.return_image = return_image

    def compute(self):
        image_test, image_true = super().compute()
        if self.mask is None:
            temp, image = self.ssim(image_test, image_true)
            self.value = torch.abs(image).mean()
        else:
            temp, image = self.ssim(image_test, image_true)
            image = torch.abs(image)
            self.value = image[:, :, self.mask[0, 0, :, :]].mean()
            self.mask = None
        self.ssim.reset()
        del temp
        # self.value = self.ssim(self.image_test * mask, self.image_true * mask)
        if self.return_image:
            return ImageTensor(torch.abs(image), permute_image=True).RGB('gray')
        else:
            return self.value

    def scale(self):
        self.range_max += self.range_max
        return self


class MultiScaleSSIM_tensor(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    def __init__(self, device):
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
        if self.mask is None:
            self.value = self.ms_ssim(image_test, image_true)
        else:
            self.value = self.ms_ssim(image_test * self.mask, image_true * self.mask)
            nb_pixel_im = image_test.shape[-2] * image_test.shape[-1]
            nb_pixel_mask = (~self.mask).to(torch.float32).sum()
            self.mask = None
            # Remove the perfect SSIM given by the mask
            self.value = (self.value * nb_pixel_im - nb_pixel_mask) / (nb_pixel_im - nb_pixel_mask)
        self.ms_ssim.reset()
        return self.value
    # def __call__(self, im1, im2, *args, mask=None, **kwargs):
    #     super().__call__(im1, im2, *args, mask=mask, **kwargs)
    #     if mask is None:
    #         self.value = self.ms_ssim(self.image_test, self.image_true)
    #     else:
    #         self.value = self.ms_ssim(self.image_test * self.mask, self.image_true * self.mask)
    #         nb_pixel_im = self.image_test.shape[-2] * self.image_test.shape[-1]
    #         nb_pixel_mask = (~self.mask).to(torch.float32).sum()
    #         # Remove the perfect SSIM given by the mask
    #         self.value = (self.value * nb_pixel_im - nb_pixel_mask) / (nb_pixel_im - nb_pixel_mask)
    #     return self.value

    def scale(self):
        return self.value


class Metric_mse_tensor(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = False

    def __init__(self, device):
        super().__init__(device)
        self.mse = MSE(squared=True).to(self.device)
        self.metric = "MSE"
        self.range_max = 1
        self.commentary = "The lower, the better"

    def update(self, preds: ImageTensor, target: ImageTensor, *args, mask=None, **kwargs) -> None:
        super().update(preds, target, *args, mask=mask, **kwargs)

    def compute(self):
        image_test, image_true = super().compute()
        if self.mask is None:
            self.value = self.mse(image_true, image_test)

        else:
            self.value = self.mse(image_true[:, :, self.mask[0, 0, :, :]].flatten(),
                                  image_test[:, :, self.mask[0, 0, :, :]].flatten())
            self.mask = None
        self.mse.reset()
        return self.value

    def scale(self):
        self.range_max = 2
        return self.value


class Metric_rmse_tensor(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = False

    def __init__(self, device):
        super().__init__(device)
        self.rmse = MSE(squared=False).to(self.device)
        self.metric = "RMSE"
        self.range_max = 1

    def update(self, preds: ImageTensor, target: ImageTensor, *args, mask=None, **kwargs) -> None:
        super().update(preds, target, *args, mask=mask, **kwargs)

    def compute(self):
        image_test, image_true = super().compute()
        if self.mask is None:
            self.value = self.rmse(image_true, image_test)
        else:
            self.value = self.rmse(image_true[:, :, self.mask[0, 0, :, :]].flatten(),
                                   image_test[:, :, self.mask[0, 0, :, :]].flatten())
            self.mask = None
        self.rmse.reset()
        return self.value

    def scale(self):
        self.value = super().scale
        return self.value


class Metric_psnr_tensor(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    def __init__(self, device):
        super().__init__(device)
        self.psnr = PSNR(data_range=None, base=10.0, reduction=None, dim=None).to(self.device)
        self.metric = "Peak Signal Noise Ratio"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = "inf"

    def update(self, preds: ImageTensor, target: ImageTensor, *args, mask=None, **kwargs) -> None:
        super().update(preds, target, *args, mask=mask, **kwargs)

    def compute(self):
        image_test, image_true = super().compute()
        try:
            if self.mask is None:
                self.value = self.psnr(image_true, image_test)
            else:
                self.value = self.psnr(image_true * self.mask[0, 0, :, :],
                                       image_test * self.mask[0, 0, :, :])
                self.mask = None
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


class Metric_nec_tensor(BaseMetric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    def __init__(self, device):
        super().__init__(device)
        self.metric = "Edges Correlation"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1

    def update(self, preds: ImageTensor, target: ImageTensor, *args, mask=None, return_image=False, **kwargs) -> None:
        super().update(preds, target, *args, mask=mask, **kwargs)
        self.return_image = return_image

    def compute(self):
        image_test, image_true = super().compute()
        ref_true = grad_tensor(ImageTensor(image_true, permute_image=True))
        ref_test = grad_tensor(ImageTensor(image_test, permute_image=True))
        if self.mask is not None:
            ref_true = ref_true * self.mask
            ref_test = ref_test * self.mask
            self.mask = None
        dot_prod = (torch.abs(torch.cos(ref_true[:, 1, :, :] - ref_test[:, 1, :, :])) *
                    ((ref_true[:, 1, :, :] != 0) + (ref_test[:, 1, :, :] != 0)))
        image_nec = (ref_true[:, 0, :, :] * ref_test[:, 0, :, :] * dot_prod)
        nec_ref = torch.sqrt(torch.sum(ref_true[:, 0, :, :] * ref_true[:, 0, :, :]) *
                             torch.sum(ref_test[:, 0, :, :] * ref_test[:, 0, :, :]))
        self.value = image_nec.sum() / nec_ref
        if self.return_image:
            return ImageTensor(image_nec, permute_image=True).RGB('gray')
        else:
            return self.value
