from pathlib import Path
from argparse import ArgumentParser
from python.plane_seg.metrics import evaluate_metric

import sys
import numpy as np


def main(argv):
    parser = ArgumentParser()
    parser.add_argument(
        "--metric_name", type=str, help="can be one of: precision, recall, mean"
    )
    parser.add_argument(
        "--print_to_console",
        help="optional, true by default, can be one of: true, false",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="optional, path to an output file.\nif specified, metric value is appended there",
    )

    args = parser.parse_args(argv)

    if not args.metric_name:
        parser.print_help()
        return

    predicted_labels = np.load("data/labels.npy")

    evaluate_metric(
        predicted_labels,
        Path("data/0000.png"),
        args.metric_name,
        print_to_console=(args.print_to_console != "false"),
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
