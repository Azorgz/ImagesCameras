import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


def len_2(vec):
    if isinstance(vec, tuple) or isinstance(vec, list) or isinstance(vec, torch.Tensor) or isinstance(vec, np.ndarray):
        return True if len(vec) == 2 else False
    else:
        return False


def scale_intrinsics(intrinsics: Float[Tensor, "*batch 3 3"], scale: Float[Tensor, "n_scale"]):
    if len(scale.shape) > 1:
        batched_scale = True
    else:
        batched_scale = False
    if intrinsics.ndim > 2:
        *batch, _, _ = intrinsics.shape
        batched = True
    else:
        intrinsics = intrinsics.unsqueeze(0)
        batch = [1]
        batched = False
    c1, c2, c3 = intrinsics.split(1, -1)
    c1 = c1 * scale
    c2 = c2 * scale
    c3 = c3.repeat(*batch, 1, c1.shape[-1])
    if not batched_scale:
        if not batched:
            return torch.stack((c1, c2, c3), dim=-2).movedim(-1, len(batch)).squeeze()
        else:
            return torch.stack((c1, c2, c3), dim=-2).movedim(-1, len(batch)).squeeze(1)
    else:
        if not batched:
            return torch.stack((c1, c2, c3), dim=-2).movedim(-1, len(batch)).squeeze(0)
        else:
            return torch.stack((c1, c2, c3), dim=-2).movedim(-1, len(batch))


def intrinsics_parameters_from_matrix(intrinsics,
                                      f=None,
                                      pixel_size=None,
                                      sensor_resolution=None,
                                      sensor_size=None,
                                      HFOV=None,
                                      VFOV=None,
                                      **kwargs) -> (np.ndarray, dict):
    assert intrinsics is not None
    if f is not None:
        f = (f / 1e3, f / 1e3) if not len_2(f) else (f[0] / 1e3, f[1] / 1e3)
        pixel_size = [f[1] / intrinsics[1, 1], f[0] / intrinsics[0, 0]]
        sensor_size = [sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1]]
        HFOV = 2 * np.arctan(sensor_size[1] / (2 * f[1])) * 180 / np.pi
        VFOV = 2 * np.arctan(sensor_size[0] / (2 * f[0])) * 180 / np.pi
        aspect_ratio = pixel_size[1] / pixel_size[0]
    elif pixel_size is not None:
        pixel_size = (pixel_size / 1e6, pixel_size / 1e6) if not len_2(pixel_size) else (
            pixel_size[0] / 1e6, pixel_size[1] / 1e6)
        f = (intrinsics[0, 0] * pixel_size[1], intrinsics[1, 1] * pixel_size[0])
        sensor_size = (sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1])
        HFOV = 2 * np.arctan(sensor_size[1] / (2 * f[0])) * 180 / np.pi
        VFOV = 2 * np.arctan(sensor_size[0] / (2 * f[1])) * 180 / np.pi
        aspect_ratio = pixel_size[1] / pixel_size[0]
    elif sensor_size is not None:
        sensor_size = (sensor_size[0] / 1e3, sensor_size[1] / 1e3)
        pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
        f = (intrinsics[0, 0] * pixel_size[1], intrinsics[1, 1] * pixel_size[0])
        HFOV = 2 * np.arctan(sensor_size[1] / (2 * f[0])) * 180 / np.pi
        VFOV = 2 * np.arctan(sensor_size[0] / (2 * f[1])) * 180 / np.pi
        aspect_ratio = pixel_size[1] / pixel_size[0]
    elif HFOV is not None and VFOV is not None:
        f = [1 / 1e3, 1 / 1e3]
        sensor_size = (2 * np.tan(VFOV / (2 * 180 / np.pi)) * f[1],
                       2 * np.tan(HFOV / (2 * 180 / np.pi)) * f[0])
        pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
        aspect_ratio = pixel_size[1] / pixel_size[0]
    elif HFOV is not None:
        f = [1 / 1e3, 1 / 1e3]
        s_size = 2 * np.tan(HFOV / (2 * 180 / np.pi)) * f[0]
        sensor_size = (s_size * sensor_resolution[0] / sensor_resolution[1], s_size)
        pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
        VFOV = 2 * np.arctan(sensor_size[0] / (2 * f[0])) * 180 / np.pi
        aspect_ratio = pixel_size[1] / pixel_size[0]
    elif VFOV is not None:
        f = [1 / 1e3, 1 / 1e3]
        s_size = 2 * np.tan(VFOV / (2 * 180 / np.pi)) * f[1]
        sensor_size = (s_size, s_size * sensor_resolution[1] / sensor_resolution[0])
        pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
        HFOV = 2 * np.arctan(sensor_size[1] / (2 * f[0])) * 180 / np.pi
        aspect_ratio = pixel_size[1] / pixel_size[0]
    else:
        f = [1 / 1e3, 1 / 1e3]
        pixel_size = [f[1] / float(intrinsics[1, 1]), f[0] / float(intrinsics[0, 0])]
        sensor_size = [sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1]]
        HFOV = 2 * np.arctan(sensor_size[1] / (2 * f[0])) * 180 / np.pi
        VFOV = 2 * np.arctan(sensor_size[0] / (2 * f[1])) * 180 / np.pi
        aspect_ratio = 1
    return intrinsics, {'f': (round(float(f[0]) * 1e3, 3) / 1e3, round(float(f[1]) * 1e3, 3) / 1e3),
                        'pixel_size': (
                            round(float(pixel_size[0]) * 1e6, 3) / 1e6, round(float(pixel_size[1]) * 1e6, 3) / 1e6),
                        'sensor_size': (
                            round(float(sensor_size[0]) * 1e3, 3) / 1e3, round(float(sensor_size[1]) * 1e3, 3) / 1e3),
                        'aspect_ratio': aspect_ratio,
                        'HFOV': round(float(HFOV), 2),
                        'VFOV': round(float(VFOV), 2)}


def intrinsics_parameters_wo_matrix(**kwargs) -> dict:
    # With f known :
    sensor_resolution = kwargs['sensor_resolution']
    if kwargs['f'] is not None:
        f = (kwargs['f'] / 1e3, kwargs['f'] / 1e3) if not len_2(kwargs['f']) else (
            kwargs['f'][0] / 1e3, kwargs['f'][1] / 1e3)
        if kwargs['pixel_size'] is not None:
            pixel_size = (kwargs['pixel_size'] / 1e6, kwargs['pixel_size'] / 1e6) if not len_2(
                kwargs['pixel_size']) else (kwargs['pixel_size'][0] / 1e6, kwargs['pixel_size'][1] / 1e6)
            sensor_size = (sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1])
            HFOV = 2 * np.arctan(sensor_size[1] / (2 * f[0])) * 180 / np.pi
            VFOV = 2 * np.arctan(sensor_size[0] / (2 * f[1])) * 180 / np.pi
            aspect_ratio = pixel_size[1] / pixel_size[0]
        elif kwargs['sensor_size'] is not None:
            sensor_size = (kwargs['sensor_size'][0] / 1e3, kwargs['sensor_size'][1] / 1e3)
            pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
            HFOV = 2 * np.arctan(sensor_size[1] / (2 * f[0])) * 180 / np.pi
            VFOV = 2 * np.arctan(sensor_size[0] / (2 * f[1])) * 180 / np.pi
            aspect_ratio = pixel_size[1] / pixel_size[0]
        elif kwargs['HFOV'] is not None and kwargs['VFOV'] is not None:
            HFOV = kwargs['HFOV']
            VFOV = kwargs['VFOV']
            sensor_size = (np.arctan(VFOV / 360 * np.pi) * 2 * f[1], np.arctan(HFOV / 360 * np.pi) * 2 * f[0])
            pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
            aspect_ratio = pixel_size[1] / pixel_size[0]
        else:
            HFOV = kwargs['HFOV']
            VFOV = kwargs['VFOV']
            sensor_size = None
            pixel_size = None
            aspect_ratio = 1

    # With the sensor size known :
    elif kwargs['pixel_size'] is not None or kwargs['sensor_size'] is not None:
        if kwargs['pixel_size'] is not None:
            pixel_size = (kwargs['pixel_size'] / 1e6, kwargs['pixel_size'] / 1e6) if not len_2(kwargs['pixel_size']) \
                else (kwargs['pixel_size'][0] / 1e6, kwargs['pixel_size'][1] / 1e6)
            sensor_size = sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1]
        elif kwargs['sensor_size'] is not None:
            sensor_size = (kwargs['sensor_size'][0] / 1e3, kwargs['sensor_size'][1] / 1e3)
            pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
        else:
            sensor_size = (kwargs['sensor_size'][0] / 1e3, kwargs['sensor_size'][1] / 1e3)
            pixel_size = (kwargs['pixel_size'] / 1e6, kwargs['pixel_size'] / 1e6) if not len_2(kwargs['pixel_size']) \
                else (kwargs['pixel_size'][0] / 1e6, kwargs['pixel_size'][1] / 1e6)
        if (kwargs['HFOV'] is not None) + (kwargs['VFOV'] is not None):
            if kwargs['HFOV'] is not None and kwargs['VFOV'] is not None:
                HFOV = kwargs['HFOV']
                VFOV = kwargs['VFOV']
                f = (
                    sensor_size[1] / (2 * np.tan(HFOV / 360 * np.pi)),
                    sensor_size[0] / (2 * np.tan(VFOV / 360 * np.pi)))
            elif kwargs['HFOV'] is not None:
                HFOV = kwargs['HFOV']
                f = (
                    sensor_size[1] / (2 * np.tan(HFOV / 360 * np.pi)),
                    sensor_size[1] / (2 * np.tan(HFOV / 360 * np.pi)))
                VFOV = 2 * np.arctan(sensor_size[1] / (2 * f[1])) * 180 / np.pi
            else:
                VFOV = kwargs['VFOV']
                f = (
                    sensor_size[0] / (2 * np.tan(VFOV / 360 * np.pi)),
                    sensor_size[0] / (2 * np.tan(VFOV / 360 * np.pi)))
                HFOV = 2 * np.arctan(sensor_size[0] / (2 * f[0])) * 180 / np.pi
        else:
            f = None
            pixel_size = kwargs['pixel_size']
            sensor_size = kwargs['sensor_size']
            HFOV = None
            VFOV = None
        aspect_ratio = pixel_size[1] / pixel_size[0]
    else:
        f = [1e-3, 1e-3]
        pixel_size = [1e-6, 1e-6]
        sensor_size = [sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1]]
        HFOV = kwargs['HFOV']
        VFOV = kwargs['VFOV']
        aspect_ratio = kwargs['aspect_ratio']
    return {'f': (round(float(f[0]) * 1e3, 3) / 1e3, round(float(f[1]) * 1e3, 3) / 1e3),
            'pixel_size': (round(float(pixel_size[0]) * 1e6, 3) / 1e6, round(float(pixel_size[1]) * 1e6, 3) / 1e6),
            'sensor_size': (round(float(sensor_size[0]) * 1e3, 3) / 1e3, round(float(sensor_size[1]) * 1e3, 3) / 1e3),
            'aspect_ratio': aspect_ratio,
            'HFOV': round(float(HFOV), 2),
            'VFOV': round(float(VFOV), 2)}
