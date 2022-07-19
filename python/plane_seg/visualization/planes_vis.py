import open3d as o3d
import numpy as np

__all__ = ["color_planes"]


def color_planes(
    pcd: o3d.geometry.PointCloud, labels: np.ndarray
) -> o3d.geometry.PointCloud:
    raise NotImplementedError
