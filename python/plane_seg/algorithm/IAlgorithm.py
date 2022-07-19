from pathlib import Path

import numpy as np
import abc

__all__ = ["IAlgorithm"]


class IAlgorithm(abc.ABC):
    @abc.abstractmethod
    def __init__(self, alg_path: Path, cfg_path: Path, pcd_path: Path):
        self.alg_path = alg_path
        self.cfg_path = cfg_path
        self.pcd_path = pcd_path

    @abc.abstractmethod
    def run(self) -> np.ndarray:
        pass
