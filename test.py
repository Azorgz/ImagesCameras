# from colormaps import *
import math
import os
from itertools import chain

import torch

from Cameras import Camera, CameraSetup
from Image import ImageTensor
from tools.drawing import extract_roi_from_images
from Cameras.base import Data

R = CameraSetup(from_file='/home/godeta/PycharmProjects/Images-Cameras/Lynred_day.yaml')

# name_path = '/'
# perso = '/home/aurelien/Images/Images_LYNRED/'
# pro = '/media/godeta/T5 EVO/Datasets/Lynred/'
#
# p = pro if 'godeta' in os.getcwd() else perso
#
# path_RGB = p + 'Day/master/visible'
# path_RGB2 = p + 'Day/slave/visible'
# path_IR = p + 'Day/master/infrared_corrected'
# path_IR2 = p + 'Day/slave/infrared_corrected'
#
# IR = Camera(path=path_IR, device=torch.device('cuda'), id='IR', f=14, name='SmartIR640', sensor_name='SmartIR640')
# IR2 = Camera(path=path_IR2, device=torch.device('cuda'), id='IR2', f=14, name='subIR', sensor_name='SmartIR640')
#
# RGB = Camera(path=path_RGB, device=torch.device('cuda'), id='RGB', f=6, name='mainRGB', pixel_size=3.45)
# RGB2 = Camera(path=path_RGB2, device=torch.device('cuda'), id='RGB2', f=6, name='subRGB', sensor_name='RGBLynred')
# R = CameraSetup(RGB, IR, IR2, RGB2, print_info=True)
#
# d_calib = 5
# center_x, center_y, center_z = 341 * 1e-03, 1, 0
#
# x, y, z = 0.127, -0.008, -0.055  # -0.0413
# rx = 0  # math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
# ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
# rz = 0
# R.update_camera_relative_position('IR', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)
#
# x, y, z = 0.127 + 0.214, 0, 0
# rx = 0  # math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
# ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
# rz = 0
# R.update_camera_relative_position('RGB2', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)
#
# x, y, z = 0.127 + 0.214 + 0.127, -0.008, -0.055
# rx = 0  # math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
# ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
# rz = 0
#
# R.update_camera_relative_position('IR2', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)
# R.calibration_for_stereo('RGB', 'RGB2')
print(len(R.cameras['RGB'].files))

# p = os.getcwd()
# path_result = p + name_path
# R.save(path_result, 'Lynred_day.yaml')
