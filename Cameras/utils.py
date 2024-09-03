import numpy as np
import torch


def len_2(vec):
    if isinstance(vec, tuple) or isinstance(vec, list) or isinstance(vec, torch.Tensor) or isinstance(vec, np.ndarray):
        return True if len(vec) == 2 else False
    else:
        return False


def intrinsics_parameters_from_matrix(**kwargs) -> (np.ndarray, dict):
    assert kwargs['intrinsics'] is not None
    intrinsics = kwargs['intrinsics'].cpu().numpy()
    sensor_resolution = kwargs['sensor_resolution']
    if kwargs['f'] is not None:
        f = (kwargs['f'] / 1e3, kwargs['f'] / 1e3) if not len_2(kwargs['f']) else (
            kwargs['f'][0] / 1e3, kwargs['f'][1] / 1e3)
        pixel_size = [f[0] / intrinsics[0, 0], f[1] / intrinsics[1, 1]]
        sensor_size = [sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1]]
        HFOV = 2 * np.arctan(sensor_size[0] / (2 * f[0])) * 180 / np.pi
        VFOV = 2 * np.arctan(sensor_size[1] / (2 * f[1])) * 180 / np.pi
        aspect_ratio = pixel_size[0] / pixel_size[1]
    elif kwargs['pixel_size'] is not None:
        pixel_size = (kwargs['pixel_size'] / 1e6, kwargs['pixel_size'] / 1e6) if not len_2(kwargs['pixel_size']) else (
            kwargs['pixel_size'][0] / 1e6, kwargs['pixel_size'][1] / 1e6)
        f = (intrinsics[0, 0] * pixel_size[0], intrinsics[1, 1] * pixel_size[1])
        sensor_size = (sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1])
        HFOV = 2 * np.arctan(sensor_size[0] / (2 * f[0])) * 180 / np.pi
        VFOV = 2 * np.arctan(sensor_size[1] / (2 * f[1])) * 180 / np.pi
        aspect_ratio = pixel_size[0] / pixel_size[1]
    elif kwargs['sensor_size'] is not None:
        sensor_size = (kwargs['sensor_size'][0] / 1e3, kwargs['sensor_size'][1] / 1e3)
        pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
        f = (intrinsics[0, 0] * pixel_size[0], intrinsics[1, 1] * pixel_size[1])
        HFOV = 2 * np.arctan(sensor_size[0] / (2 * f[0])) * 180 / np.pi
        VFOV = 2 * np.arctan(sensor_size[1] / (2 * f[1])) * 180 / np.pi
        aspect_ratio = pixel_size[0] / pixel_size[1]
    else:
        f = [1, 1]
        pixel_size = [f[0] / float(intrinsics[0, 0]), f[1] / float(intrinsics[1, 1])]
        sensor_size = [sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1]]
        HFOV = kwargs['HFOV']
        VFOV = kwargs['VFOV']
        aspect_ratio = kwargs['aspect_ratio']
    return intrinsics, {'f': (round(f[0] * 1e3, 3) / 1e3, round(f[1] * 1e3, 3) / 1e3),
                        'pixel_size': (round(pixel_size[0] * 1e6, 3) / 1e6, round(pixel_size[1] * 1e6, 3) / 1e6),
                        'sensor_size': (round(sensor_size[0] * 1e3, 3) / 1e3, round(sensor_size[1] * 1e3, 3) / 1e3),
                        'aspect_ratio': aspect_ratio,
                        'HFOV': round(HFOV, 2),
                        'VFOV': round(VFOV, 2)}


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
            aspect_ratio = pixel_size[0] / pixel_size[1]
        elif kwargs['sensor_size'] is not None:
            sensor_size = (kwargs['sensor_size'][0] / 1e3, kwargs['sensor_size'][1] / 1e3)
            pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
            HFOV = 2 * np.arctan(sensor_size[1] / (2 * f[0])) * 180 / np.pi
            VFOV = 2 * np.arctan(sensor_size[0] / (2 * f[1])) * 180 / np.pi
            aspect_ratio = pixel_size[0] / pixel_size[1]
        elif kwargs['HFOV'] is not None and kwargs['VFOV'] is not None:
            HFOV = kwargs['HFOV']
            VFOV = kwargs['VFOV']
            sensor_size = (np.arctan(HFOV / 360 * np.pi) * 2 * f[0], np.arctan(VFOV / 360 * np.pi) * 2 * f[1])
            pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
            aspect_ratio = pixel_size[0] / pixel_size[1]
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
                    sensor_size[0] / (2 * np.tan(HFOV / 360 * np.pi)),
                    sensor_size[1] / (2 * np.tan(VFOV / 360 * np.pi)))
            elif kwargs['HFOV'] is not None:
                HFOV = kwargs['HFOV']
                f = (
                    sensor_size[0] / (2 * np.tan(HFOV / 360 * np.pi)),
                    sensor_size[0] / (2 * np.tan(HFOV / 360 * np.pi)))
                VFOV = 2 * np.arctan(sensor_size[1] / (2 * f[1])) * 180 / np.pi
            else:
                VFOV = kwargs['VFOV']
                f = (
                    sensor_size[1] / (2 * np.tan(VFOV / 360 * np.pi)),
                    sensor_size[1] / (2 * np.tan(VFOV / 360 * np.pi)))
                HFOV = 2 * np.arctan(sensor_size[0] / (2 * f[0])) * 180 / np.pi
        else:
            f = None
            pixel_size = kwargs['pixel_size']
            sensor_size = kwargs['sensor_size']
            HFOV = None
            VFOV = None
            aspect_ratio = kwargs['aspect_ratio']
        aspect_ratio = pixel_size[0] / pixel_size[1]
    else:
        f = [1, 1]
        pixel_size = [1, 1]
        sensor_size = [sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1]]
        HFOV = kwargs['HFOV']
        VFOV = kwargs['VFOV']
        aspect_ratio = kwargs['aspect_ratio']
    return {'f': (round(f[0] * 1e3, 3) / 1e3, round(f[1] * 1e3, 3) / 1e3),
            'pixel_size': (round(pixel_size[0] * 1e6, 3) / 1e6, round(pixel_size[1] * 1e6, 3) / 1e6),
            'sensor_size': (round(sensor_size[0] * 1e3, 3) / 1e3, round(sensor_size[1] * 1e3, 3) / 1e3),
            'aspect_ratio': aspect_ratio,
            'HFOV': round(HFOV, 2),
            'VFOV': round(VFOV, 2)}
