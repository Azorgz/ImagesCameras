# from colormaps import *
import os

from kornia.color import rgb_to_lab

from Image import ImageTensor
from tools.drawing import extract_roi_from_images

i = ImageTensor(os.getcwd() + '/vis.png')
i[:, :, :-100, :-100] = 0
i[:, :, 500:510, 450:470] = 0
mask = extract_roi_from_images(i)
mask.show()



