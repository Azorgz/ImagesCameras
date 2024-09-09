import inspect
import math
import warnings
import cv2
import imagesize
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from types import FrameType
from typing import cast, Union
from kornia.geometry import PinholeCamera, axis_angle_to_rotation_matrix, transform_points, depth_to_3d_v2, \
    rotation_matrix_to_axis_angle
from torch import Tensor, nn
from torch.nn import MaxPool2d

from ..Image import ImageTensor, DepthTensor
from .Sensors import load_sensor
from .base import Data, Sensor
from ..tools.misc import print_tuple
from .utils import intrinsics_parameters_wo_matrix, intrinsics_parameters_from_matrix

ext_available = ['/*.png', '/*.jpg', '/*.jpeg', '/*.tif', '/*.tiff']


class Camera(PinholeCamera):
    """
    A camera object subclassing the camera from Kornia.
    :param path: Always required (valid path or in the future stream url)
    :param device: not mandatory (defined as gpu if a gpu is available)
    :param name: not mandatory, defined as 'BaseCam' by default
    :param id: not mandatory, defined as 'BaseCam' by default
    :param is_positioned: not mandatory, defined as False by default. This parameter is True if the relative pose of the camera in the camera rig is known
    :param aperture: only cosmetic parameter for now
    :param intrinsics: if define it will create a camera with this matrix. If no other parameters is specified, the FOV/f/pixel size/sensor size will be defined by default
    :param f: focal in milimeters, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as sensor_size or pixel_size...
    :param pixel_size: pixel size in micrometer, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as sensor_size or focal...
    :param sensor_size: sensor size in milimeters, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as focal or pixel_size...
    :param HFOV: horizontal angular field of view in degrees
    :param VFOV: vertical angular field of view in degrees
    :param sensor_resolution: defined by the images find in path resolution. If set it's only a control value emetting a warning if the source images are not at matching resolution
    :param extrinsics: if define the camera will be positionned using that matrix, ignoring the following parameters. If no pose parameter is given, the pose will be canonical by default
    :param x: relative position in x defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param y: relative position in y defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param z: relative position in y defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param rx: relative rotation around x defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param ry: relative rotation around y defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param rz: relative rotation around z defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param in_degree: set to True if you express the position angles in degrees
    """
    _is_positioned = False
    _is_ref = False
    _setup = None
    _name = 'BaseCam'
    _id = 'BaseCam'
    _f = None
    _VFOV = None
    _HFOV = None

    def __init__(self,
                 # Data source
                 path: (str or Path) = None,
                 fps=30,
                 # General parameters
                 device=None,
                 name='BaseCam',
                 id='BaseCam',
                 sensor_name=None,
                 is_positioned=False,
                 # Intrinsic args
                 intrinsics: Union[Tensor, np.ndarray] = None,
                 f: (float or tuple) = None,
                 pixel_size: (float or tuple) = None,
                 sensor_size: tuple = None,
                 HFOV: float = None,
                 VFOV: float = None,
                 aspect_ratio: float = 1.0,
                 sensor_resolution: tuple = None,
                 # Extrinsic args
                 extrinsics: Union[Tensor, np.ndarray] = None,
                 x: float = None,
                 y: float = None,
                 z: float = None,
                 rx: float = None,
                 ry: float = None,
                 rz: float = None,
                 in_degree: bool = False,
                 **kwargs) -> None:

        # General parameters #################################################################
        self._id = id
        self._name = name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = Data(path, fps)

        # Intrinsics parameters definition #################################################################
        h, w, modality, im_calib = self._init_resolution_()
        self._modality = modality
        self._im_calib = im_calib
        # Sensor
        if sensor_name is not None and sensor_name != 'NoName':
            self.sensor = load_sensor(sensor_name)
            intrinsics, parameters = self._init_intrinsics_(
                intrinsics=intrinsics,
                f=f,
                pixel_size=self.sensor.pixelSize,
                sensor_size=self.sensor.size,
                HFOV=HFOV,
                VFOV=VFOV,
                aspect_ratio=self.sensor.aspect_ratio,
                sensor_resolution=self.sensor.resolution)
        else:
            sensor_resolution = sensor_resolution if sensor_resolution is not None else (h, w)
            intrinsics, parameters = self._init_intrinsics_(
                intrinsics=intrinsics,
                f=f,
                pixel_size=pixel_size,
                sensor_size=sensor_size,
                HFOV=HFOV,
                VFOV=VFOV,
                aspect_ratio=aspect_ratio,
                sensor_resolution=sensor_resolution)
            self.sensor = Sensor(parameters['pixel_size'], parameters['sensor_size'], sensor_resolution, modality)
        if self.sensor.resolution[1] != w or self.sensor.resolution[0] != h:
            warnings.warn(
                f'The specified sensor resolution {self.sensor.resolution} and the source image resolution {(h, w)} are different')

        self._f = parameters['f']
        self._VFOV = parameters['VFOV']
        self._HFOV = parameters['HFOV']

        # Extrinsic parameters definition #################################################################
        if extrinsics is None:
            self.is_positioned = False
            extrinsics = self._init_extrinsics_(x, y, z, rx, ry, rz)
        else:
            extrinsics = torch.tensor(extrinsics, dtype=torch.double).unsqueeze(0).to(self.device)
            self.is_positioned = is_positioned

        h = torch.tensor(self.sensor_resolution[0]).unsqueeze(0).to(self.device)
        w = torch.tensor(self.sensor_resolution[1]).unsqueeze(0).to(self.device)
        super(Camera, self).__init__(intrinsics, extrinsics, h, w)

    def display_src(self):
        for im in self.data():
            image = ImageTensor(im).to_opencv()
            cv2.imshow(f'Camera {self.name} : {self.data.fps}fps', image)
            cv2.waitKey(int(30 / self.data.fps))

    def __str__(self):
        optical_parameter = self.optical_parameter()
        global_parameter = {i: j for i, j in self.save_dict().items() if i not in optical_parameter}
        string1 = '\n'.join([': '.join([str(key), str(v)]) for key, v in global_parameter.items()])
        gap = "\n\noptical parameters : \n".upper()
        string2 = '\n'.join([': '.join([str(key), print_tuple(v)]) for key, v in optical_parameter.items()])
        return string1 + gap + string2

    def optical_parameter(self):
        return {'f': (self.f[0] * 10 ** 3, "mm"),
                'HFOV': (self.HFOV, '°'),
                'VFOV': (self.VFOV, '°')} | self.sensor.repr()

    def save_dict(self):
        return {'name': self.name,
                'id': self.id,
                'path': self.path,
                'intrinsics': self.intrinsics.squeeze().cpu().numpy().tolist(),
                'extrinsics': self.extrinsics.squeeze().cpu().numpy().tolist(),
                'is_ref': self.is_ref,
                'is_positioned': self.is_positioned,
                'f': [self.f[0] * 1e3, self.f[1] * 1e3]} | self.sensor.save_dict()

    def _init_resolution_(self) -> tuple:
        im_path = ''
        im_size = None
        for f in self.data():
            if im_size is None:
                im_size = imagesize.get(f)
            else:
                assert im_size == imagesize.get(f), \
                    f'Several images size have been found, image_size enforced at {im_size}'
            if 'calibration_image' in f:
                if im_path == '':
                    im_path = f
        if im_path == '':
            im_path = self.data(0)[0]
        im_calib = ImageTensor(im_path, device=self.device)
        _, c, h, w = im_calib.shape
        modality = im_calib.modality
        return h, w, modality, im_calib

    def _init_intrinsics_(self, **kwargs):
        if kwargs['intrinsics'] is not None:
            intrinsics, parameters = intrinsics_parameters_from_matrix(**kwargs)
            intrinsics = torch.tensor(intrinsics, dtype=torch.double).unsqueeze(0).to(self.device)
        elif not all([v is None for k, v in list(kwargs.items()) if k != 'aspect_ratio' and k != 'sensor_resolution']):
            parameters = intrinsics_parameters_wo_matrix(**kwargs)
            intrinsics = self._init_intrinsics_matrix(kwargs['sensor_resolution'][1], kwargs['sensor_resolution'][0],
                                                      parameters['f'], parameters['pixel_size'], None)
        else:
            intrinsics = self._init_intrinsics_matrix(kwargs['sensor_resolution'][1], kwargs['sensor_resolution'][0],
                                                      None, None, None)[0]
            kwargs['intrinsics'] = intrinsics
            intrinsics, parameters = intrinsics_parameters_from_matrix(**kwargs)
        return intrinsics, parameters

    def _init_intrinsics_matrix(self, h: Union[int, None], w: Union[int, None], f: Union[tuple, list, None],
                                pixel_size: Union[tuple, list, None], center: Union[tuple, list, None],
                                **kwargs) -> Tensor:
        """
        :param h: Height of the images (required if center is None, otherwise ignored)
        :param w: Width of the images (required if center is None, otherwise ignored)
        :param f: Focal length (required if w, h is None)
        :param pixel_size: Pixel size of the images (only used if f is not None)
        :param center: Center of the images (required if w, h is None)
        :return: Init the intrinsic matrix of the camera with default parameter
        """
        if f is not None and pixel_size is not None:
            d = f[0] / pixel_size[0], f[1] / pixel_size[1]
        elif f is not None:
            d = f[0], f[1]
        else:
            d = (np.sqrt(h ** 2 + w ** 2), np.sqrt(h ** 2 + w ** 2))
        if center is not None:
            cx = center[0]
            cy = center[1]
        else:
            cx = (w / 2)
            cy = (h / 2)
        return torch.tensor([[d[0], 0, cx, 0],
                             [0, d[1], cy, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=torch.double).unsqueeze(0).to(self.device)

    def _init_extrinsics_(self, x, y, z, rx, ry, rz, **kwargs) -> Tensor:
        """ :return: init the camera in the space at the position O(0, 0, 0) wo rotation """
        x = x if x is not None else 0
        y = y if y is not None else 0
        z = z if z is not None else 0
        mat_tr = Tensor([x, y, z, 1])
        rx = rx if rx is not None else 0
        ry = ry if ry is not None else 0
        rz = rz if rz is not None else 0
        mat_rot = axis_angle_to_rotation_matrix(Tensor([rx, ry, rz]).unsqueeze(0))
        mat = torch.zeros([1, 4, 4])
        mat[:, :3, :3] = mat_rot
        mat[:, :, -1] = mat_tr
        return mat.inverse().to(dtype=torch.double).to(self.device)

    def reset(self):
        """Only settable by the _del_camera_ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '_del_camera_':
            self.is_ref = False
            self.is_positioned = False

    def update_setup(self, camera_ref, cameras) -> None:
        self.setup = cameras
        self.is_ref = self.id == camera_ref
        self.is_positioned = True if self.is_ref else self.is_positioned

    def update_id(self, idx) -> None:
        setattr(self, 'id', f'{self.id}_{idx}')

    def update_pos(self, extrinsics=None, x=None, y=None, z=None, x_pix=None, y_pix=None, rx=None, ry=None, rz=None):
        x = x if x is not None else (
            x_pix * self.pixel_size[0] / self.f[0] if x_pix is not None else self.extrinsics[0, 0, 3].cpu())
        y = y if y is not None else (
            y_pix * self.pixel_size[1] / self.f[1] if y_pix is not None else self.extrinsics[0, 1, 3].cpu())
        if extrinsics is None:
            self.extrinsics = self._init_extrinsics_(x, y, z, rx, ry, rz)
        else:
            self.extrinsics = extrinsics
        self.is_positioned = True

    def pixel_size_at(self, distance=0):
        """
        :param distance: distance of interest or list of a point of interest coordinates
        :return: The size of a pixel in the image at such a distance
        """
        if isinstance(distance, list) or isinstance(distance, tuple):
            if len(distance) == 3:
                distance = Tensor(distance).to(self.device)
                distance = torch.sqrt(torch.sum((distance - self.center) ** 2))
        if not distance:
            distance = torch.sqrt(torch.sum(self.center ** 2))
        fov_v = 2 * math.tan(self.VFOV / 360 * math.pi) * distance
        fov_h = 2 * math.tan(self.HFOV / 360 * math.pi) * distance
        return fov_h / self.width, fov_v / self.height

    def __getitem__(self, index, autopad=False, **kwargs):
        im_path = self.files[index]
        im = ImageTensor(im_path, device=self.device)
        if autopad:
            temp = torch.zeros(*im.shape[:-2], self.sensor_resolution[1], self.sensor_resolution[0])
            im = im.pad(temp)
        return im

    def random_image(self, autopad=False, **kwargs):
        # list_im = os.listdir(self.path))
        index = torch.randint(0, len(self.files), [1])
        im = self.__getitem__(index, autopad=autopad)
        return im, index

    def depth_to_3D(self, depth: DepthTensor, image_texture: Tensor = None, euclidian_depth=False, **kwargs):
        # Project the points in a 3D space according the camera intrinsics
        points_3D: Tensor = depth_to_3d_v2(depth[:, 0, :, :], self.camera_matrix[0], normalize_points=euclidian_depth)
        # Camera frame to world frame
        points_3D = transform_points(self.extrinsics.inverse()[:, None], points_3D.to(torch.float64))
        # If texture is given, the final point-cloud will be of shape (b x h x w x 6)
        if image_texture is not None:
            assert depth.shape[-2:] == image_texture.shape[-2:]
            points_3D = torch.stack([points_3D, image_texture.permute([0, 2, 3, 1])], dim=3)
        return points_3D

    def project_world_to_image(self, pointcloud: Tensor, image=None, level=1,
                               from_world_frame=True) -> ImageTensor or DepthTensor:
        """
        :param from_world_frame: If True, project first the Pointcloud within the Camera frame
        :param pointcloud: 3D point cloud in the World frame coordinate system
        :param image: image to texture the cloud with. Has to be the same size as the cloud. If None a depth image is printed
        :param level: projection gaussian pyramid level
        :return: ImageTensor
        """
        if from_world_frame:
            pointcloud = transform_points(self.extrinsics[:, None], pointcloud.to(torch.float64))
        cloud_in_camera_frame = torch.concatenate([self.project(pointcloud), pointcloud[:, :, :, -1:]], dim=3)
        if image is not None:
            assert cloud_in_camera_frame.shape[1:3] == image.shape[-2:]
            b, cha, h, w = image.shape
            image_flatten = image.reshape([b, cha, w * h])  # shape b x c x H*W
            projectedImage = ImageTensor(torch.zeros([b, cha, self.sensor_resolution[1],
                                                      self.sensor_resolution[0]])).to(dtype=pointcloud.dtype)
            sample = image_flatten.to(pointcloud.dtype)
        else:
            cha = 1
            b = pointcloud.shape[0]
            projectedImage = DepthTensor(torch.zeros([b, cha, self.sensor_resolution[1],
                                                      self.sensor_resolution[0]])).to(dtype=pointcloud.dtype)
        # Put all the point into a H*W x 3 vector
        c = cloud_in_camera_frame.reshape((cloud_in_camera_frame.shape[0],
                                           cloud_in_camera_frame.shape[1] *
                                           cloud_in_camera_frame.shape[2], 3))  # B x H*W x 3
        # Sort the point by decreasing depth
        indexes = torch.argsort(c[..., -1], descending=True)
        c_ = c.clone()
        c_sorted = c_.clone()
        for j in range(b):
            c_sorted[j, ...] = c_[j, indexes[j], :]
        if image is not None:
            for j in range(b):
                sample[j] = sample[j, :, indexes[j]]
        else:
            sample = c_sorted[..., 2:]
        for i in reversed(range(level)):
            im_size = self.sensor_resolution[1] // (2 ** i), self.sensor_resolution[0] // (2 ** i)
            c_ = c_sorted.clone()
            # Normalize the point cloud over the sensor resolution
            c_[:, :, 0] /= 2 ** i
            c_[:, :, 1] /= 2 ** i

            # Transform the landing positions in accurate pixels
            c_x = torch.round(c_[..., 0:1]).to(torch.int32)
            c_y = torch.round(c_[..., 1:2]).to(torch.int32)
            # Remove the point landing outside the image
            mask_out = ((c_x < 0) + (c_x >= im_size[1]) +
                        (c_y < 0) + (c_y >= im_size[0])) != 0
            c_x[mask_out] = 0
            c_y[mask_out] = 0
            # rays = np.concatenate([c_x, c_y], axis=2)
            image_at_level = torch.zeros([b, cha, im_size[0], im_size[1]], dtype=pointcloud.dtype).to(pointcloud.device)
            image_at_level[:, :, c_y.squeeze(), c_x.squeeze()] = sample
            temp = F.interpolate(image_at_level, projectedImage.shape[-2:])
            projectedImage[temp > 0] = temp[temp > 0]
        conv_upsampling = MaxPool2d((3, 5), stride=1, padding=(1, 2), dilation=1)
        projectedImage[projectedImage == 0] = conv_upsampling(projectedImage)[projectedImage == 0]
        return projectedImage

    @property
    def center(self):
        return Tensor([self.tx, self.ty, self.tz]).to(self.device)

    @property
    def f(self):
        return self._f

    @property
    def path(self):
        return self.data.path

    @path.setter
    def path(self, value):
        self.data.update_path(value)

    @property
    def pixel_size(self):
        return self.sensor.pixelSize

    @property
    def sensor_size(self):
        return self.sensor.size

    @property
    def sensor_resolution(self):
        return self.sensor.resolution

    @property
    def aspect_ratio(self):
        return self.sensor.aspect_ratio

    @property
    def modality(self):
        return self.sensor.modality.modality

    @property
    def private_modality(self):
        return self.sensor.modality.private_modality

    @property
    def VFOV(self):
        return self._VFOV

    @property
    def HFOV(self):
        return self._HFOV

    @property
    def pixel_FOV(self):
        return self.VFOV / self.sensor_resolution[0], self.HFOV / self.sensor_resolution[1]

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def files(self, *args):
        return self.data(*args)

    @property
    def extrinsics(self):
        return self._extrinsics

    @extrinsics.setter
    def extrinsics(self, value):
        """Only settable by the __init__, __new__, update_pos methods"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '__new__' or name == 'update_pos' or name == '__init__':
            if not isinstance(value, Tensor):
                value = Tensor(value)
            if value.device != self.device:
                value = value.to(self.device)
            if value.dtype != torch.float64:
                value = torch.tensor(value, dtype=torch.double)
            if value.shape != torch.Size([1, 4, 4]):
                value = value.unsqueeze(0)
            self._extrinsics = value

    @property
    def setup(self):
        return self._setup

    @setup.setter
    def setup(self, setup):
        """Only settable by the update_rig method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == 'update_setup' or name == '_reset_':
            self._setup = setup

    @setup.deleter
    def setup(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def im_calib(self):
        return self._im_calib

    @im_calib.setter
    def im_calib(self, im_calib):
        """Only settable by the __init__ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '__init__':
            self._im_calib = im_calib

    @im_calib.deleter
    def im_calib(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def is_ref(self):
        return self._is_ref

    @is_ref.setter
    def is_ref(self, is_ref):
        """Only settable by the __init__ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == 'update_camera_ref' or name == 'reset':
            self._is_ref = is_ref

    @is_ref.deleter
    def is_ref(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def is_positioned(self):
        return self._is_positioned

    @is_positioned.setter
    def is_positioned(self, is_positioned):
        """Only settable by the _register_camera_ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == 'register_camera' or name == '_reset_' or '__init__' or '__new__':
            self._is_positioned = is_positioned

    @is_positioned.deleter
    def is_positioned(self):
        warnings.warn("The attribute can't be deleted")


class LearnableCamera(Camera, nn.Module):

    def __init__(self, *args, **kwargs):
        Camera.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        rotation_angles = rotation_matrix_to_axis_angle(self._extrinsics[:, :3, :3].inverse()).cpu()
        translation_vector = self._extrinsics[:, :3, 3].cpu()
        fx, fy = self._extrinsics[:, 0, 0], self._extrinsics[:, 1, 1]
        cx, cy = self._extrinsics[:, 0, 2], self._extrinsics[:, 1, 2]
        self._fx, self._fy = fx, fy
        self._cx, self._cy = cx, cy
        self._intrinsics = self._init_intrinsics_matrix(None, None, (fx, fy), None, (cx, cy))
        self._rotation_angles = nn.Parameter(rotation_angles, requires_grad=True)
        self._translation_vector = nn.Parameter(translation_vector, requires_grad=True)
        self.update_pos(rx=self._rotation_angles[0, 0],
                        ry=self._rotation_angles[0, 1],
                        rz=self._rotation_angles[0, 2],
                        x=self._translation_vector[0, 0],
                        y=self._translation_vector[0, 1],
                        z=self._translation_vector[0, 2])

    @property
    def extrinsics(self):
        return self._extrinsics

    @extrinsics.setter
    def extrinsics(self, value):
        """Only settable by the __init__, __new__, update_pos methods"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '__new__' or name == 'update_pos' or name == '__init__':
            if (self.rotation_angles is None or self.translation_vector is None) and value is None:
                if not isinstance(value, Tensor):
                    value = Tensor(value)
                if value.device != self.device:
                    value = value.to(self.device)
                if value.dtype != torch.float64:
                    value = torch.tensor(value, dtype=torch.double)
                if value.shape != torch.Size([1, 4, 4]):
                    value = value.unsqueeze(0)
                self._extrinsics = value
            else:
                self.update_pos(rx=self._rotation_angles[0, 0],
                                ry=self._rotation_angles[0, 1],
                                rz=self._rotation_angles[0, 2],
                                x=self._translation_vector[0, 0],
                                y=self._translation_vector[0, 1],
                                z=self._translation_vector[0, 2])

    @property
    def rotation_angles(self):
        return self._rotation_angles

    @rotation_angles.setter
    def rotation_angles(self, value):
        self._rotation_angles = nn.Parameter(value, requires_grad=True)
        self.update_pos(rx=self._rotation_angles[0, 0],
                        ry=self._rotation_angles[0, 1],
                        rz=self._rotation_angles[0, 2],
                        x=self._translation_vector[0, 0],
                        y=self._translation_vector[0, 1],
                        z=self._translation_vector[0, 2])

    @property
    def translation_vector(self):
        return self._translation_vector

    @translation_vector.setter
    def translation_vector(self, value):
        self._translation_vector = nn.Parameter(value, requires_grad=True)
        self.update_pos(rx=self._rotation_angles[0, 0],
                        ry=self._rotation_angles[0, 1],
                        rz=self._rotation_angles[0, 2],
                        x=self._translation_vector[0, 0],
                        y=self._translation_vector[0, 1],
                        z=self._translation_vector[0, 2])

    @property
    def f(self):
        return self.fx, self.fy

    @f.setter
    def f(self, value):
        if len(value) == 2:
            fx, fy = value[0], value[1]
        else:
            fx, fy = value
        self.fx = fx
        self.fy = fy

    @property
    def fx(self):
        return self._fx

    @fx.setter
    def fx(self, value):
        self._fx = nn.Parameter(value, requires_grad=True)
        self.intrinsics = 0

    @property
    def fy(self):
        return self._fy

    @fy.setter
    def fy(self, value):
        self._fy = nn.Parameter(value, requires_grad=True)
        self.intrinsics = 0

    @property
    def cx(self):
        return self._cx

    @cx.setter
    def cx(self, value):
        self._cx = nn.Parameter(value, requires_grad=True)
        self.intrinsics = 0

    @property
    def cy(self):
        return self._cy

    @cy.setter
    def cy(self, value):
        self._cy = nn.Parameter(value, requires_grad=True)
        self.intrinsics = 0

    @property
    def intrinsics(self):
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, value):
        self._intrinsics = self._init_intrinsics_matrix(None, None, (self.fx, self.fy), None, (self.cx, self.cy))
