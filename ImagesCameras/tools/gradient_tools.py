import cv2 as cv
import numpy as np
import torch
from kornia import pi
from kornia.color import hsv_to_rgb
from torch import Tensor
from torchmetrics.functional.image import image_gradients
from torchvision.transforms.functional import gaussian_blur

from ..Image import ImageTensor


def normalisation_tensor(image):
    m, M = image.min(), image.max()
    if m != M:
        return (image - m) / (M - m)
    else:
        return image


def grad_image(image: ImageTensor) -> ImageTensor:
    image = image.GRAY().squeeze().cpu().numpy()
    Ix = cv.Sobel(image, cv.CV_64F, 1, 0, borderType=cv.BORDER_REFLECT_101)
    Iy = cv.Sobel(image, cv.CV_64F, 0, 1, borderType=cv.BORDER_REFLECT_101)
    grad = np.sqrt(Ix ** 2 + Iy ** 2)
    grad[grad < grad.mean()] = 0
    grad[grad > grad.mean() * 5] = grad.mean() * 5
    orient = cv.phase(Ix, Iy, angleInDegrees=True)
    orient[grad == 0] = 0

    v = cv.normalize(grad, None, 0, 255, cv.NORM_MINMAX)
    s = np.ones_like(grad) * 255
    s[grad == 0] = 0
    h = cv.normalize(orient % 180, None, 0, 255, cv.NORM_MINMAX)
    h[grad == 0] = 0
    output = np.uint8(np.stack([h, s, v], axis=-1))
    output = cv.cvtColor(output, cv.COLOR_HSV2RGB)
    return ImageTensor(output)


def grad_tensor_image(image_tensor: ImageTensor, device=None) -> ImageTensor:
    im_t = image_tensor.put_channel_at(1)
    c = image_tensor.shape[1]
    ratio = torch.sum(image_tensor > 0)/(image_tensor.shape[3]*image_tensor.shape[2])
    im_t = gaussian_blur(im_t, [5, 5])
    dy, dx = image_gradients(im_t)
    if c > 1:
        dx, dy = torch.sum(dx, dim=1) / 3, torch.sum(dy, dim=1) / 3
    grad_im = torch.sqrt(dx ** 2 + dy ** 2)
    m = torch.mean(grad_im)
    grad_im[grad_im < m * 2 / ratio] = 0
    grad_im[grad_im > m * 5] = 5*m
    # kernel = torch.ones(3, 3).to(device)
    # grad_im = dilation(grad_im.unsqueeze(0), kernel).squeeze(0)
    mask = grad_im == 0
    orient = torch.atan2(dy, dx)  # / np.pi * 180
    orient[mask] = 0
    v = normalisation_tensor(grad_im)
    s = torch.ones_like(grad_im)
    s[mask] = 0
    h = normalisation_tensor(orient % pi) * (2 * pi)
    h[mask] = 0
    output = hsv_to_rgb(torch.stack([h, s, v], dim=1).squeeze(2))
    return output.to(device)


def grad_tensor(image_tensor) -> Tensor:
    """
    A differentiable version of the grad tensor function
    """
    im_t = image_tensor.put_channel_at(1).to_tensor()
    ratio = torch.nonzero(im_t).sum() / torch.mul(*im_t.shape[-2:])  # Ratio of non zeros pixels
    dy, dx = image_gradients(im_t)  # Gradient dx dy
    # dx, dy = gaussian_blur(dx, [3, 1]), gaussian_blur(dy, [1, 3])
    grad_im = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)  # Amplitude of the gradient
    m = torch.mean(grad_im)  # Mean of the grad Amplitude
    grad_im = torch.clamp(grad_im, max=5 * m)  # Ceil all value to a maximum of 5*m
    grad_im = torch.clamp(grad_im, min=2 * m / ratio)  # Floor all value to a minimum of 2*m / ratio of pixels != 0
    mask = (grad_im != 2 * m / ratio)*1.  # All the pixel (> 2 * m / ratio) are valid
    orient = torch.atan2(dy + 1e-8, dx + 1e-8)*mask  # / np.pi * 180
    grad_im, arg_grad = grad_im.max(dim=1, keepdims=True)  # Takes the most informative channel
    grad_im = normalisation_tensor(grad_im)
    return torch.cat([grad_im, orient.gather(1, arg_grad)], dim=1)
