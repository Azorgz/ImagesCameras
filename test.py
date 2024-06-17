# from colormaps import *
import os

from kornia.color import rgb_to_lab

from Image import ImageTensor

i = ImageTensor(os.getcwd() + '/vis.png')
i.show()
i_hsv = i.HSV()
i_hsv.RGB().show()
i_luv = i_hsv.LUV()
i_luv.RGB().show()
i_lab = i_luv.LAB()
i_lab.RGB().show()
i = i.RGB()
# i.RGB().show()
i.show()

