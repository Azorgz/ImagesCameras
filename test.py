# from colormaps import *
import math
import os
from itertools import chain

import torch

from Cameras import Camera, CameraSetup
from Image import ImageTensor
from tools.drawing import extract_roi_from_images
from Cameras.base import Data
from Vizualisation import Visualizer

Visualizer("/home/godeta/PycharmProjects/Disparity_Pipeline/results/Dataset_Lynred").run()
