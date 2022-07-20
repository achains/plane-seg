import sys
from pathlib import Path

from argparse import ArgumentParser
import numpy as np

from . import evaluate_metrics


def main(argv):
    parser = ArgumentParser()
    parser.add_argument('--predicted_labels', type=Path, help='path to a .npy file')
    parser.add_argument('--ground_truth', type=Path, help='path to a .png or a .npy file')
    parser.add_argument('--metric_name', type=str, help='can be one of: precision, recall, mean')
    parser.add_argument('--print_to_console', help='true by default, can be one of: true, false')
    parser.add_argument('--output_file', type=Path,
                        help='path to an output file.\nif specified, metric value is appended there')

    args = parser.parse_args(argv)

    if not args.predicted_labels or not args.ground_truth or not args.metric_name:
        parser.print_help()
        return

    predicted_labels = np.load(args.predicted_labels)

    evaluate_metrics(predicted_labels,
                     args.ground_truth,
                     args.metric_name,
                     print_to_console=(args.print_to_console != 'false'),
                     output_file=args.output_file)


if __name__ == '__main__':
    main(sys.argv[1:])
