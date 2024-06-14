# from colormaps import *

from kornia.color import rgb_to_lab

from Image import ImageTensor

i = ImageTensor.rand()
lab1 = rgb_to_lab(i)
lab = i.LAB()
(lab - lab1).show()
