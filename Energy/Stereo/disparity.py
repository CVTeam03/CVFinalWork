from stereo.SSD import disparity as disparity_ssd
from stereo.GC import disparity as disparity_gc

SSD = 'ssd'
GRAPHCUT = 'graphcut'

DISPARITY_METHODS = {
    SSD: disparity_ssd,
    GRAPHCUT: disparity_gc,
}


def disparity(image_left, image_right, **kwargs):
    method = kwargs.pop('method')
    disparity_method = DISPARITY_METHODS.get(method, lambda: None)

    return disparity_method(image_left, image_right, **kwargs)