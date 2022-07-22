from typing import AnyStr, Dict, Collection
from pathlib import Path
from evops.metrics import *
from . import rgb2labels

import cv2

__all__ = ["evaluate_metrics"]


def evaluate_metrics(
    prediction: NDArray[(Any, Any), np.int32],
    ground_truth_path: Path,
    metric_names: Collection[AnyStr] = (
        "iou",
        "dice",
        "precision-iou",
        "recall-iou",
        "fScore-iou",
        "mean-iou",
        "mean-dice",
    ),
    print_to_console: bool = True,
    output_file: Path = None,
) -> Dict[AnyStr, np.float64]:
    """
    Evaluates an EVOPS metric given predictions and ground truth

    :param prediction: an RGB image or a 2D NumPy array
    :param ground_truth_path: path to an RGB image or a 2D NumPy array, the same size as predictions
    :param metric_names: names of EVOPS metrics: any subset of: {iou, dice, precision-iou, recall-iou, fScore-iou, mean-iou, mean-dice}. All by default
    :param print_to_console: prints to console if true
    :param output_file: appends output to the file if specified
    :return: the value of the metric
    """
    if not ground_truth_path.exists():
        raise ValueError("ground truth file not found")
    if ground_truth_path.suffix not in (".png", ".npy"):
        raise ValueError("ground truth may only be an RGB image or a NumPy array")

    if ground_truth_path.suffix == ".png":
        ground_truth = rgb2labels.rgb2labels(cv2.imread(str(ground_truth_path)))
    else:
        ground_truth = np.load(ground_truth_path)

    metric_values = {}

    if "iou" in metric_names:
        metric_values["iou"] = iou(prediction, ground_truth)
    if "dice" in metric_names:
        metric_values["dice"] = dice(prediction, ground_truth)
    if "precision-iou" in metric_names:
        metric_values["precision-iou"] = precision(prediction, ground_truth, "iou")
    if "recall-iou" in metric_names:
        metric_values["recall-iou"] = recall(prediction, ground_truth, "iou")
    if "fScore-iou" in metric_names:
        metric_values["fScore-iou"] = fScore(prediction, ground_truth, "iou")
    if "mean-iou" in metric_names:
        metric_values["mean-iou"] = mean(prediction, ground_truth, iou)
    if "mean-dice" in metric_names:
        metric_values["mean-dice"] = mean(prediction, ground_truth, dice)

    if print_to_console:
        for metric, value in metric_values.items():
            print(f"{metric:13} {value}")
    if output_file is not None:
        with open(output_file, "a") as fout:
            for metric, value in metric_values.items():
                fout.write(f"{metric:13} {value}\n")
            fout.write("\n")

    return metric_values
