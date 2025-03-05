import itertools
import os
from dataclasses import dataclass
from glob import glob
from itertools import chain
from typing import Iterable

from ..Image.base import Modality


@dataclass()
class Sensor:
    r"""Data class to represent the camera sensor.

    Args:
        pixelSize: float | Iterable[float, float], micrometer size of a pixel (h, w) or u
        size: Iterable[float, float], size in milimiters of the sensor (h, w)
        resolution: Iterable[int, int]
        modality: str

    Example:
        >>> sensor = Sensor(16, [4, 5], [500, 600], 'Infrared')
    """
    pixelSize: list[float]
    size: list[float]
    resolution: list[int]
    modality: Modality
    name: str = 'NoName'

    def __init__(self, pixelSize: float | Iterable[float],
                 size: Iterable[float],
                 resolution: Iterable[int],
                 modality: Modality,
                 sensor_name: str = None):
        self.pixelSize = [*pixelSize] if isinstance(pixelSize, Iterable) else [pixelSize, pixelSize]
        self.size = list(size)
        self.resolution = list(resolution)
        self.modality = Modality(modality)
        if sensor_name is not None:
            self.name = sensor_name

    def repr(self):
        return {
            'pixel_size': ((self.pixelSize[0] * 10 ** 6, self.pixelSize[1] * 10 ** 6), 'um'),
            'sensor_size': ((self.size[0] * 10 ** 3, self.size[1] * 10 ** 3), 'mm'),
            'width': (self.resolution[1], 'pixels'),
            'height': (self.resolution[0], 'pixels')}

    def save_dict(self):
        return {
            'pixelSize': self.pixelSize,
            'size': self.size,
            'resolution': self.resolution,
            'modality': self.modality.modality,
            'sensor_name': self.name}

    @property
    def aspect_ratio(self):
        return self.pixelSize[0] / self.pixelSize[1]


class Data:
    r"""Data class to represent the data produced by the camera.

    Args:
        path : str | path | stream url
        fps : int
        file : Generator

    Example:
        >>> data = data('path', 30)
        >>> data.path
        'path'
        >>> data.fps
        30
    """
    path: str
    fps: int
    ext_available = ['/*.png', '/*.jpg', '/*.jpeg', '/*.tif', '/*.tiff']

    def __init__(self, path, fps, list_file=None):
        self.path = path
        self.fps = fps
        if list_file is None:
            if isinstance(path, list):
                gen = []
                for p in path:
                    gen.extend(self._init_path(p))
                self.generator = gen
            else:
                self.generator = list(self._init_path(path))
        else:
            self.generator = list_file

    def _init_path(self, path):
        try:
            assert os.path.exists(path)

            def path_generator(p):
                for ext in self.ext_available:
                    for f in sorted(glob(p + ext)):
                        yield f

            return path_generator(path)

        except AssertionError:
            print(f'There is no image file at {path}, a source is necessary to configure a new camera')
            raise AssertionError

    def update_path(self, path):
        self.path = path
        self.reset_generator()

    def reset_generator(self):
        if isinstance(self.path, list):
            gen = []
            for p in self.path:
                gen.append(self._init_path(p))
            self.generator = list(chain(*gen))
        elif self.path is not None:
            self.generator = list(self._init_path(self.path))

    def __call__(self, *args, **kwargs):
        if not args:
            return self.generator
        else:
            gen = []
            self.reset_generator()
            for idx in args:
                gen.append(self.generator[idx])
            return list(gen)
