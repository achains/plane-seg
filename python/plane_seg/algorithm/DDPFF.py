from pathlib import Path
from shutil import rmtree

from . import IAlgorithm
from . import Config

import subprocess
import os

import numpy as np
import open3d as o3d

__all__ = ["DDPFF"]


class DDPFF(IAlgorithm.IAlgorithm):
    def __init__(self, alg_path: Path, cfg_path: Path, pcd_path: Path):
        self.alg_path = alg_path
        self.cfg_path = cfg_path
        self.pcd_path = pcd_path
        self.__alg_input_dir = Path("ddpff_input")
        self.__alg_output_dir = Path("ddpff_output")
        self.__alg_artifact_name = Path("planes.txt")
        self.__parameter_list = (
            "debugLevel",
            "bufferSize",
            "floodFill.pointThreshold_min",
            "floodFill.pointThreshold_max",
            "floodFill.planeThreshold_flood",
            "floodFill.planeThreshold_merge",
            "floodFill.planeThreshold_flood_max",
            "floodFill.planeThreshold_merge_max",
            "floodFill.angleThresholdFloodFill",
            "floodFill.angleThresholdFloodFill_max",
            "floodFill.minPlaneSize",
            "floodFill.normalSampleDistance_min",
            "floodFill.normalSampleDistance_max",
            "floodFill.c_plane",
            "floodFill.c_plane_merge",
            "floodFill.c_point",
            "floodFill.c_angle",
            "floodFill.c_range",
        )

    def run(self) -> np.ndarray:
        if os.path.exists(self.__alg_output_dir):
            rmtree(self.__alg_output_dir)
        if os.path.exists(self.__alg_input_dir):
            rmtree(self.__alg_input_dir)

        os.mkdir(self.__alg_output_dir)
        os.mkdir(self.__alg_input_dir)

        # .pcd/.ply -> .ply in ascii
        pcd_path = DDPFF.__preprocess_pcd(
            pcd_path=self.pcd_path, output_dir=self.__alg_input_dir
        )

        subprocess.run(
            [
                self.alg_path,
                pcd_path,
                self.cfg_path,
                self.__alg_output_dir / self.__alg_artifact_name,
            ]
        )

        try:
            labels = DDPFF.__alg_output_to_labels(
                self.__alg_output_dir / self.__alg_artifact_name, self.pcd_path
            )
        finally:
            rmtree(self.__alg_output_dir)
            rmtree(self.__alg_input_dir)

        return labels

    @staticmethod
    def __preprocess_pcd(pcd_path: Path, output_dir: Path) -> Path:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pcd.paint_uniform_color([0, 0, 0])
        pcd_path = str(output_dir / pcd_path.stem) + ".ply"
        o3d.io.write_point_cloud(pcd_path, pcd, write_ascii=True)
        return Path(pcd_path)

    @staticmethod
    def __alg_output_to_labels(artifact_path: Path, pcd_path: Path) -> np.ndarray:
        planes = []
        with open(artifact_path) as f:
            for line in f:
                planes.append(np.asarray([int(x) for x in line.split()]))

        number_of_points = np.asarray(
            o3d.io.read_point_cloud(str(pcd_path)).points
        ).shape[0]
        labels = np.zeros(number_of_points, dtype=int)
        s = set()
        for index, plane_indices in enumerate(planes):
            col = np.random.uniform(0, 1, size=(1, 3))

            while tuple(col[0]) in s:
                col = np.random.uniform(0, 1, size=(1, 3))

            s.add(tuple(col[0]))

            if plane_indices.size > 0:
                labels[plane_indices] = index + 1

        return labels
