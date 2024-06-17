# from colormaps import *
import os

from kornia.color import rgb_to_lab

from Image import ImageTensor

i = ImageTensor(os.getcwd() + '/vis.png')
i = i.GRAY()
i = i.RGB('viridis')
i.show()
