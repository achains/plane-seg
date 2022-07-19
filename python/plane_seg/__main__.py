import numpy as np

from algorithm.DDPFF import DDPFF
from pathlib import Path

import argparse
import sys


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default="labels")

    args = parser.parse_args(argv)

    algorithm = DDPFF(args.algorithm, args.config, args.data)
    labels = algorithm.run()
    np.save(args.output, labels)


if __name__ == "__main__":
    main(sys.argv[1:])
