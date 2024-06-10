import os
from dataclasses import dataclass
from glob import glob
from itertools import chain


@dataclass()
class Sensor:
    r"""Data class to represent the camera sensor.

    Args:
        pixelSize: int | tuple[int, int], micrometer size of a pixel (h, w) or u
        size: tuple[int, int], size in milimiters of the sensor (h, w)
        resolution: tuple[int, int]
        modality: str

    Example:
        >>> sensor = Sensor(16, (4, 5), (500, 600), 'Infrared')
    """
    pixelSize: int | tuple[int, int]
    size: tuple[int, int]
    resolution: tuple[int, int]
    modality: str


@dataclass()
class data:
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

    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        if isinstance(path, list):
            gen = []
            for p in path:
                gen.extend(self._init_path(p))
            self.generator = chain(*gen)
        else:
            self.generator = self._init_path(path)

    def _init_path(self, path):
        try:
            assert os.path.exists(path)

            def path_generator(p):
                for ext in self.ext_available:
                    for f in glob(p + ext):
                        yield f

            return path_generator(path)

        except AssertionError:
            print(f'There is no image file at {path}, a source is necessary to configure a new camera')
            raise AssertionError

    def __call__(self, *args, **kwargs):
        return self.generator.__next__()
