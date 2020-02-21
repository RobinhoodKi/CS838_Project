import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

MAX_ITER = 10
POS_W = 9
POS_XY_STD = 3
Bi_W = 10
Bi_XY_STD = 90
Bi_RGB_STD = 5


def dense_crf(img, p_map):
    c = p_map.shape[0]
    h = p_map.shape[1]
    w = p_map.shape[2]

    U = utils.unary_from_softmax(p_map)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    label_map = np.argmax(Q, axis=0)
    return label_map