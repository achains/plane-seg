from typing import AnyStr, Dict
from pathlib import Path
from evops.metrics import *
from . import rgb2labels

import cv2

__all__ = ["evaluate_metric", "evaluate_all_metrics"]


def evaluate_metric(
    prediction: NDArray[(Any, Any), np.int32],
    ground_truth_path: Path,
    metric_name: AnyStr,
    print_to_console: bool = True,
    output_file: Path = None,
) -> np.float64:
    """
    Evaluates an EVOPS metric given predictions and ground truth

    :param prediction: an RGB image or a 2D NumPy array
    :param ground_truth_path: path to an RGB image or a 2D NumPy array, the same size as predictions
    :param metric_name: name of an EVOPS metric: may be one of: {iou, dice, precision-iou, recall-iou, fScore-iou, mean-iou, mean-dice}
    :param print_to_console: prints to console if true
    :param output_file: appends output to the file if specified
    :return: the value of the metric
    """
    if not ground_truth_path.exists():
        raise ValueError("ground truth file not found")
    if ground_truth_path.suffix not in (".png", ".npy"):
        raise ValueError("ground truth may only be an RGB image or a NumPy array")
    if metric_name not in (
        "iou",
        "dice",
        "precision-iou",
        "recall-iou",
        "fScore-iou",
        "mean-iou",
        "mean-dice",
    ):
        raise ValueError(
            "invalid metric. must be one of: {iou, dice, precision-iou, recall-iou, fScore-iou, mean-iou, mean-dice}"
        )

    if ground_truth_path.suffix == ".png":
        ground_truth = rgb2labels.rgb2labels(cv2.imread(str(ground_truth_path)))
    else:
        ground_truth = np.load(ground_truth_path)

    if metric_name == "iou":
        metric_value = iou(prediction, ground_truth)
    elif metric_name == "dice":
        metric_name = dice(prediction, ground_truth)
    elif metric_name == "precision-iou":
        metric_value = precision(prediction, ground_truth, "iou")
    elif metric_name == "recall-iou":
        metric_value = recall(prediction, ground_truth, "iou")
    elif metric_name == "fScore-iou":
        metric_value = fScore(prediction, ground_truth, "iou")
    elif metric_name == "mean-iou":
        metric_value = mean(prediction, ground_truth, iou)
    else:
        metric_value = mean(prediction, ground_truth, dice)

    if print_to_console:
        print(f"{metric_name}: {metric_value}")
    if output_file is not None:
        with open(output_file, "a") as fout:
            fout.write(f"{metric_name}: {metric_value}\n")

    return metric_value


def evaluate_all_metrics(
    prediction: NDArray[(Any, Any), np.int32],
    ground_truth_path: Path,
    print_to_console: bool = True,
    output_file: Path = None,
) -> Dict[AnyStr, np.float64]:
    """
    Evaluates all EVOPS metrics given predictions and ground truth

    :param prediction: an RGB image or a 2D NumPy array
    :param ground_truth_path: path to an RGB image or a 2D NumPy array, the same size as predictions
    :param print_to_console: prints to console if true
    :param output_file: appends output to the file if specified
    :return: dictionary where keys are metric names and values are metric values
    """
    if not ground_truth_path.exists():
        raise ValueError("ground truth file not found")
    if ground_truth_path.suffix not in (".png", ".npy"):
        raise ValueError("ground truth may only be an RGB image or a NumPy array")

    if ground_truth_path.suffix == ".png":
        ground_truth = rgb2labels.rgb2labels(cv2.imread(str(ground_truth_path)))
    else:
        ground_truth = np.load(ground_truth_path)

    all_metrics_values = {
        "iou": iou(prediction, ground_truth),
        "dice": dice(prediction, ground_truth),
        "precision-iou": precision(prediction, ground_truth, "iou"),
        "recall-iou": recall(prediction, ground_truth, "iou"),
        "fScore-iou": fScore(prediction, ground_truth, "iou"),
        "mean-iou": mean(prediction, ground_truth, iou),
        "mean-dice": mean(prediction, ground_truth, dice),
    }

    if print_to_console:
        for metric, value in all_metrics_values.items():
            print(f"{metric:13} {value}")
    if output_file is not None:
        with open(output_file, "a") as fout:
            for metric, value in all_metrics_values.items():
                fout.write(f"{metric:13} {value}\n")
            fout.write("\n")

    return all_metrics_values
