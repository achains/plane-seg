from typing import Union, Any, Callable

from PIL import Image
import numpy as np
from nptyping import NDArray

from . import rgb2labels

__all__ = ["get_metric"]


def get_metric(gtdata: Union[Image.Image, NDArray[Any, np.int32]],
               preddata: Union[Image.Image, NDArray[Any, np.int32]],
               metric: Callable[[NDArray[Any, np.int32], NDArray[Any, np.int32]], np.float64]) -> np.float64:
    if isinstance(gtdata, Image.Image):
        gtdata = rgb2labels.rgb2labels(gtdata)
    if isinstance(preddata, Image.Image):
        preddata = rgb2labels.rgb2labels(preddata)

    return metric(gtdata, preddata)
