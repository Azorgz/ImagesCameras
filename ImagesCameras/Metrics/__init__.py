from __future__ import annotations
import torch
from .Metrics import SSIM, MultiScaleSSIM, MSE, RMSE, PSNR, PSNR, NEC, GradientCorrelation, SCC, NCC

# Version variable
__version__ = "1.0"

norms_dict = {'rmse': RMSE,
              'psnr': PSNR,
              'ssim': SSIM,
              'ms_ssim': MultiScaleSSIM,
              'nec': NEC,
              'gc': GradientCorrelation,
              'scc': SCC,
              'mse': MSE,
              'NCC': NCC}

stats_dict = {'mean': torch.mean,
              'std': torch.std}

__all__ = ["SSIM",
           "MultiScaleSSIM",
           "MSE",
           "RMSE",
           "PSNR",
           "NEC",
           "SCC",
           "NCC",
           'GradientCorrelation',
           "norms_dict",
           "stats_dict"]
