from ..Image.utils import in_place_fct
from ..Image import ImageTensor


def draw_rectangle(im: ImageTensor, roi: list, in_place=True):
    out = in_place_fct(im, in_place)
    temp = out.to_numpy()

