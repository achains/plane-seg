from typing import Any

import numpy as np
from PIL import Image
from nptyping import NDArray, Int


def rgb2labels(image: Image.Image) -> NDArray[(Any, Any), Int]:
    assert (image.mode == 'RGB'), "not an RGB image"
    image_arr = np.array(image)
    reshaped = image_arr.reshape([image_arr.shape[0] * image_arr.shape[1], image_arr.shape[2]])
    labeled = reshaped[0] * 255**2 + reshaped[1] * 255 + reshaped[2]
    return labeled
