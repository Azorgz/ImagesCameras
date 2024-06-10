# from colormaps import *
from itertools import chain


# vis = ImageTensor(os.getcwd() + "/vis.png")
# vis[:, :, :, 900:] = 0
# vis[:, :, 800:, 700:] = 0
# vis2 = ImageTensor.rand(1, 3).match_shape(vis)
# vis2[:, :, :100, :200] = 0
# vis2.show()
#
# res = extract_roi_from_images(vis)
# fus = (vis + vis2)/2
# fus.show(roi=[res[0], res[1]])
# # vis = vis2.LAB()
# # vis.show()
# vis2.permute('h', 0, 'w', 'channel', in_place=True)
# vis2.pprint()
# RGBA = RGBA_to_GRAY()

def generator1():
    for i in range(10):
        yield i


def generator2():
    for i in range(10):
        yield i + 100


def generator3():
    for i in range(10):
        yield i + 1000


def generator4():
    return chain(generator1(), generator2(), generator3())


for i in generator4():
    print(i)
