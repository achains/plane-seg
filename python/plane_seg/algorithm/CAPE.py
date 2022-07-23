from pathlib import Path
from shutil import rmtree
from docker.types import Mount

from typing import Collection

from . import Algorithm

import docker

import numpy as np
import open3d as o3d

__all__ = ["CAPE"]


class CAPE(Algorithm.Algorithm):
    def __init__(self, container_name: str, cfg_path: Path, pcd_path: Path):
        self.container_name = container_name
        self.cfg_path = cfg_path
        self.pcd_path = pcd_path
        self._cfg = None
        self._alg_input_dir = Path("cape_input")
        self._alg_output_dir = Path("cape_output")
        self._alg_artifact_name = Path("labels_0.csv")
        self._parameter_list = (
            "depthSigmaCoeff",
            "depthSigmaMargin",
            "cylinderScoreMin",
            "cylinderRansacSqrMaxDist",
            "cosAngleMax",
            "maxMergeDist",
            "patchSize",
            "minNrOfValidPointsOnePerXThreshold",
            "planesegMaxDiff",
            "planarFittingJumpsCounterThresholdParam",
            "histogramBinsPerCoordParam",
            "regionGrowingCandidateSizeThresholdParam",
            "regionGrowingCellsActivatedThresholdParam",
            "regionPlanarFittingPlanarityScoreThresholdParam",
            "cylinderDetectionCellsActivatedThreshold",
            "refinementMultiplierParam"
        )

    # getting parameters for running algo
    def _preprocess_input(self) -> Collection[str]:
        pcd_name = self.__convert_point_cloud_to_depth_image().name
        cfg_name = self._cfg.write(self._alg_input_dir / "params.ini").name

        return [pcd_name, cfg_name]

    def __convert_point_cloud_to_depth_image(self) -> Path:
        pcd = o3d.io.read_point_cloud(str(self.pcd_path))
        pcd.paint_uniform_color([0, 0, 0])

        xyz_load = np.asarray(pcd.points)
        z = xyz_load[:, 2].reshape(480, 640)  # TODO: get rid of magic numbers
        d = (z * 1000).astype(np.uint32)
        img = o3d.geometry.Image(d.astype(np.uint16))

        img_path = str(self._alg_input_dir / "depth_0.png")
        o3d.io.write_image(img_path, img)

        return Path(img_path)

    def _evaluate_algorithm(self, input_parameters: Collection[str]) -> Path:
        client = docker.from_env()
        input_mount = Mount(
            target="/app/build/input",
            source=str(self._alg_input_dir.absolute()),
            type="bind",
        )
        output_mount = Mount(
            target="/app/build/output",
            source=str(self._alg_output_dir.absolute()),
            type="bind",
        )

        client.containers.run(
            self.container_name,
            " ".join(input_parameters),
            mounts=[input_mount, output_mount],
        )

        return self._alg_output_dir / self._alg_artifact_name

    def _output_to_labels(self, output_path: Path) -> np.ndarray:
        labels_table = np.genfromtxt(output_path, delimiter=',').astype(np.uint8)
        labels = labels_table.reshape(labels_table.size)
        return labels

    def _clear_artifacts(self):
        rmtree(self._alg_input_dir)
        rmtree(self._alg_output_dir)
