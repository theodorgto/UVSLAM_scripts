#!/usr/bin/python3
import os
import time
import argparse
import collections
import numpy
import small_gicp
import open3d
from pyridescence import *
import pandas as pd

def crop_box(pcd, xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Keep only points within the axis-aligned box [xmin,xmax]×[ymin,ymax]×[zmin,zmax].
    """
    pts = numpy.asarray(pcd.points)
    mask = (
        (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
        (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
        (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
    )
    indices = numpy.where(mask)[0]
    return pcd.select_by_index(indices)

# Odometry estimation based on scan-to-scan matching
class ScanToScanMatchingOdometry(object):
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.T_last_current = numpy.identity(4)
        self.T_world_lidar = numpy.identity(4)
        self.target = None

    def estimate(self, raw_points):
        downsampled, tree = small_gicp.preprocess_points(raw_points, 0.25, num_threads=self.num_threads)

        if self.target is None:
            self.target = (downsampled, tree)
            return self.T_world_lidar

        result = small_gicp.align(
            self.target[0],
            downsampled,
            self.target[1],
            self.T_last_current,
            num_threads=self.num_threads
        )

        self.T_last_current = result.T_target_source
        self.T_world_lidar = self.T_world_lidar @ result.T_target_source
        self.target = (downsampled, tree)

        return self.T_world_lidar


# Odometry estimation based on scan-to-model matching
class ScanToModelMatchingOdometry(object):
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.T_last_current = numpy.identity(4)
        self.T_world_lidar = numpy.identity(4)
        self.target = small_gicp.GaussianVoxelMap(1.0)
        self.target.set_lru(horizon=100, clear_cycle=10)

    def estimate(self, raw_points):
        downsampled, tree = small_gicp.preprocess_points(raw_points, 0.25, num_threads=self.num_threads)

        if self.target.size() == 0:
            self.target.insert(downsampled)
            return self.T_world_lidar

        result = small_gicp.align(
            self.target,
            downsampled,
            self.T_world_lidar @ self.T_last_current,
            num_threads=self.num_threads
        )

        self.T_last_current = numpy.linalg.inv(self.T_world_lidar) @ result.T_target_source
        self.T_world_lidar = result.T_target_source
        self.target.insert(downsampled, self.T_world_lidar)

        return self.T_world_lidar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_threads', help='Number of threads', type=int, default=4)
    parser.add_argument('-m', '--model', help='Use scan-to-model matching odometry', action='store_true')
    args = parser.parse_args()

    dataset_path = 'data/recording'
    filenames = sorted([dataset_path + '/' + x for x in os.listdir(dataset_path) if x.endswith('.ply')])

    step_size = 5
    start_idx = 100
    points = 300
    filenames = filenames[start_idx:start_idx + points:step_size]

    if not args.model:
        odom = ScanToScanMatchingOdometry(args.num_threads)
    else:
        odom = ScanToModelMatchingOdometry(args.num_threads)

    time_queue = collections.deque(maxlen=500)
    poses = []

    # This list will accumulate every scan (in world frame)
    all_transformed = []

    for i, filename in enumerate(filenames):
        raw_points_o3d = open3d.io.read_point_cloud(filename)

        # Filter out far-away points (e.g. z ≥ 1.3 m)
        # distance_threshold = 1.3
        # pts = numpy.asarray(raw_points_o3d.points)
        # mask = pts[:, 2] < distance_threshold
        # raw_points_o3d = raw_points_o3d.select_by_index(numpy.where(mask)[0])
        # raw_points = numpy.asarray(raw_points_o3d.points)

        # bounds
        XMIN, XMAX = -0.5, 0.6
        YMIN, YMAX = -1.0, 1.0
        ZMIN, ZMAX = -0.3, 1.0

        raw_points_o3d = crop_box(raw_points_o3d, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
        raw_points = numpy.asarray(raw_points_o3d.points)

        print(f"Processing {filename} ({i + 1}/{len(filenames)})")

        t1 = time.time()
        T = odom.estimate(raw_points)  # 4×4 transform from lidar to world
        t2 = time.time()

        poses.append(T[:3, 3].copy())

        # Transform this scan into the world frame:
        num_pts = raw_points.shape[0]
        homog = numpy.ones((num_pts, 4), dtype=raw_points.dtype)
        homog[:, :3] = raw_points  # (N, 4)
        transformed_homog = (T @ homog.T).T  # (N, 4)
        transformed_points = transformed_homog[:, :3]  # (N, 3)

        all_transformed.append(transformed_points)
        time_queue.append(t2 - t1)

    # After processing all scans, stack them into one big point cloud
    if len(all_transformed) == 0:
        print("No scans were processed—exiting.")
        return

    map_points = numpy.vstack(all_transformed)  # (sum_of_all_Ni, 3)
    print(f"Total points in map: {map_points.shape[0]}")

    # Convert to Open3D point cloud and save
    map_o3d = open3d.geometry.PointCloud()
    map_o3d.points = open3d.utility.Vector3dVector(map_points)
    out_filename = "map.ply"
    open3d.io.write_point_cloud(out_filename, map_o3d)
    print(f"Saved accumulated map to '{out_filename}'")

    # OPTIONAL: Also save trajectory (XYZ) as CSV
    traj = numpy.array(poses)  # shape = (num_scans, 3)
    traj_df = pd.DataFrame(traj, columns=["X", "Y", "Z"])
    traj_df.to_csv("trajectory.csv", index=False)
    print("Saved trajectory to 'trajectory.csv'")

if __name__ == '__main__':
    main()
