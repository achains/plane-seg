from typing import Union, Any, Callable
from pathlib import Path

import cv2
import numpy as np
from nptyping import NDArray

from . import rgb2labels

__all__ = ["evaluate_metrics"]


def evaluate_metrics(prediction: Union[NDArray[(Any, Any, 3), np.int32], NDArray[(Any, Any), np.int32]],
                     ground_truth_path: Path,
                     metric: Callable[[NDArray[Any, np.int32], NDArray[Any, np.int32]], np.float64],
                     print_to_console: bool = True,
                     output_file: Path = None) -> np.float64:
    """
    Evaluates an EVOPS metric given predictions and ground truth

    :param prediction: an RGB image or a 2D NumPy array
    :param ground_truth_path: path to an RGB image or a 2D NumPy array, the same size as predictions
    :param metric: a callable EVOPS metric
    :param print_to_console: prints to console if true
    :param output_file: appends output to the file if specified
    :return: the value of the metric
    """
    assert ground_truth_path.exists(), "ground truth file not found"
    assert ground_truth_path.suffix in ('.png', '.npy'), "ground truth may only be an RGB image or a NumPy array"

    if len(prediction.shape) == 3:
        prediction = rgb2labels.rgb2labels(prediction)

    if ground_truth_path.suffix == '.png':
        ground_truth = rgb2labels.rgb2labels(cv2.imread(ground_truth_path))
    else:
        ground_truth = np.load(ground_truth_path)

    metric_value = metric(prediction, ground_truth)

    if print_to_console:
        print(metric_value)
    if output_file is not None:
        with open(output_file, 'a') as fout:
            fout.write(f'{metric_value}\n')

    return metric_value
