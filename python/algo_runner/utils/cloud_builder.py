import os
import numpy as np
import open3d as o3d

from pathlib import Path

__all__ = ["labels_to_cloud"]


def labels_to_cloud(plane_labels_path: Path, raw_cloud: Path, output_path: Path):
    pcd = o3d.io.read_point_cloud(str(raw_cloud))

    planes = []
    with open(plane_labels_path) as f:
        for line in f:
            planes.append(np.asarray([int(x) for x in line.split()]))

    colors = np.array(pcd.colors)
    labels = np.zeros(colors.shape[0], dtype=int)
    s = set()
    for index, plane_indices in enumerate(planes):

        col = np.random.uniform(0, 1, size=(1, 3))

        while tuple(col[0]) in s:
            col = np.random.uniform(0, 1, size=(1, 3))

        s.add(tuple(col[0]))

        if plane_indices.size > 0:
            colors[plane_indices] = col
            labels[plane_indices] = index + 1

    pcd.colors = o3d.utility.Vector3dVector(colors)

    np.save(os.path.join(output_path, "{}.npy".format("labels")), labels)
    o3d.io.write_point_cloud(
        os.path.join(output_path, "{}.pcd".format("labeled_pc")), pcd
    )
