from python.plane_seg.algorithm import DDPFF
from pathlib import Path

import sys


def main(argv):
    executable_path = Path(argv[0])
    algorithm = DDPFF(
        alg_path=executable_path,
        cfg_path=Path("data/ddpff.ini"),
        pcd_path=Path("data/0.ply"),
    )

    labels = algorithm.run()
    print(labels)


if __name__ == "__main__":
    main(sys.argv[1:])
