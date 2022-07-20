from typing import Union, Any, Callable, AnyStr
from pathlib import Path

import cv2
import numpy as np
from nptyping import NDArray
from evops.metrics import *

from . import rgb2labels

__all__ = ["evaluate_metrics"]


def evaluate_metrics(prediction: Union[NDArray[(Any, Any, 3), np.int32], NDArray[(Any, Any), np.int32]],
                     ground_truth_path: Path,
                     metric_name: AnyStr,
                     # condition_name: AnyStr,
                     print_to_console: bool = True,
                     output_file: Path = None) -> np.float64:
    """
    Evaluates an EVOPS metric given predictions and ground truth

    :param prediction: an RGB image or a 2D NumPy array
    :param ground_truth_path: path to an RGB image or a 2D NumPy array, the same size as predictions
    :param metric_name: name of an EVOPS metric: can be one of: {iou, dice, precision-iou, recall-iou, fScore-iou, mean-iou, mean-dice}
    :param print_to_console: prints to console if true
    :param output_file: appends output to the file if specified
    :return: the value of the metric
    """
    assert ground_truth_path.exists(), "ground truth file not found"
    assert ground_truth_path.suffix in ('.png', '.npy'), "ground truth may only be an RGB image or a NumPy array"
    assert metric_name in ('iou', 'dice', 'precision-iou', 'recall-iou', 'fScore-iou', 'mean-iou', 'mean-dice'), \
        "invalid metric"

    if len(prediction.shape) == 3:
        prediction = rgb2labels.rgb2labels(prediction)

    if ground_truth_path.suffix == '.png':
        ground_truth = rgb2labels.rgb2labels(cv2.imread(str(ground_truth_path)))
    else:
        ground_truth = np.load(ground_truth_path)

    if metric_name == 'iou':
        metric_value = iou(prediction, ground_truth)
    elif metric_name == 'dice':
        metric_name = dice(prediction, ground_truth)
    elif metric_name == 'precision-iou':
        metric_value = precision(prediction, ground_truth, 'iou')
    elif metric_name == 'recall-iou':
        metric_value = recall(prediction, ground_truth, 'iou')
    elif metric_name == 'fScore-iou':
        metric_value = fScore(prediction, ground_truth, 'iou')
    elif metric_name == 'mean-iou':
        metric_value = mean(prediction, ground_truth, iou)
    else:
        metric_value = mean(prediction, ground_truth, dice)

    if print_to_console:
        print(f'{metric_name}: {metric_value}')
    if output_file is not None:
        with open(output_file, 'a') as fout:
            fout.write(f'{metric_name}: {metric_value}\n')

    return metric_value
