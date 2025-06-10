#!/usr/bin/python3
import os
import time
import argparse
import collections
import numpy
import small_gicp
import open3d
from pyridescence import *
import matplotlib.pyplot as plt
import pandas as pd

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

        result = small_gicp.align(self.target[0], downsampled, self.target[1], self.T_last_current, num_threads=self.num_threads)
        
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
        
        result = small_gicp.align(self.target, downsampled, self.T_world_lidar @ self.T_last_current, num_threads=self.num_threads)

        self.T_last_current = numpy.linalg.inv(self.T_world_lidar) @ result.T_target_source
        self.T_world_lidar = result.T_target_source
        self.target.insert(downsampled, self.T_world_lidar)
        
        guik.viewer().update_drawable('target', glk.create_pointcloud_buffer(self.target.voxel_points()[:, :3]), guik.Rainbow())
        
        return self.T_world_lidar

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

def main():
    parser = argparse.ArgumentParser()
    #   parser.add_argument('dataset_path', help='/path/to/kitti/velodyne')
    parser.add_argument('--num_threads', help='Number of threads', type=int, default=4)
    parser.add_argument('-m', '--model', help='Use scan-to-model matching odometry', action='store_true')
    args = parser.parse_args()
  
    # dataset_path = args.dataset_path
    dataset_path = 'data/recording'
    filenames = sorted([dataset_path + '/' + x for x in os.listdir(dataset_path) if x.endswith('.ply')])

    step_size = 5
    points = 1000
    start_idx = 0

    filenames = filenames[start_idx:points:step_size]  
    
    if not args.model:
        odom = ScanToScanMatchingOdometry(args.num_threads)
    else:
        odom = ScanToModelMatchingOdometry(args.num_threads)
    
    viewer = guik.viewer()
    viewer.disable_vsync()
    time_queue = collections.deque(maxlen=500)
    poses = []

    for i, filename in enumerate(filenames):
        # raw_points = numpy.fromfile(filename, dtype=numpy.float32).reshape(-1, 4)[:, :3]
        raw_points_o3d = open3d.io.read_point_cloud(filename)
        # raw_points = numpy.asarray(raw_points_o3d.points)

        # distance_threshold = 1.3
        # raw_points_o3d = raw_points_o3d.select_by_index(numpy.where(numpy.asarray(raw_points_o3d.points)[:, 2] < distance_threshold)[0])
        # raw_points = numpy.asarray(raw_points_o3d.points)



        # bounds
        XMIN, XMAX = -0.5, 0.6
        YMIN, YMAX = -1.0, 1.0
        ZMIN, ZMAX = -0.3, 1.0

        raw_points_o3d = crop_box(raw_points_o3d, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
        raw_points = numpy.asarray(raw_points_o3d.points)



        print(f"Processing {filename}")

        t1 = time.time()
        T = odom.estimate(raw_points)
        t2 = time.time()

        # print(T[:3, 3])
        # print(T)

        poses.append(T[:3, 3].copy())

    
        time_queue.append(t2 - t1)
        viewer.lookat(T[:3, 3])
        viewer.update_drawable('points', glk.create_pointcloud_buffer(raw_points), guik.FlatOrange(T).add('point_scale', 2.0))
        
        if i % 10 == 0:
            viewer.update_drawable('pos_{}'.format(i), glk.primitives.coordinate_system(), guik.VertexColor(T))
            viewer.append_text('avg={:.3f} msec/scan  last={:.3f} msec/scan'.format(1000 * numpy.mean(time_queue), 1000 * time_queue[-1]))
            
        if not viewer.spin_once():
            break    
    

    # load the CSV file
    df_depth = pd.read_csv('data/bluerov2_mavros_global_position_rel_alt.csv')
    df_pcl = pd.read_csv('data/recording/frame_timestamps.csv')
    
    # find time range of pcl data
    start_time = df_pcl['left_timestamp'].min()
    end_time = df_pcl['left_timestamp'].max()
    # print(f"length of depth data: {len(df_depth)}")
    
    # only use depth data that is within the time range of the pcl data
    df_depth = df_depth[(df_depth['timestamp'] >= start_time) & (df_depth['timestamp'] <= end_time)]

    # apply the same step size to the timestamps
    pcl_timestamps = df_pcl['left_timestamp'].values[0:points:step_size]

    # print(f"Number of poses: {len(poses)}")
    # print(f"Number of depth values: {len(df_depth['value'])}")

  

    # plot the trajectory
    poses = numpy.array(poses)
    plt.figure(figsize=(10, 6))
    plt.plot(pcl_timestamps,-poses[:, 0], label='X Position')
    plt.plot(pcl_timestamps,-poses[:, 1], label='Y Position')
    plt.plot(pcl_timestamps,-poses[:, 2], label='Z Position')
    plt.plot(df_depth['timestamp'], df_depth['value'], label='Depth Relative Altitude', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Timestamp (ns)')
    plt.ylabel('Position')
    plt.title('Odometry Trajectory')
    plt.savefig('trajectory.png')
    plt.show()

if __name__ == '__main__':
    main()