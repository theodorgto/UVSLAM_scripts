#!/usr/bin/env python3
"""
visualize_masked_pointclouds.py

This script loads point clouds and corresponding masks, applies each mask to the point cloud,
and visualizes the original and masked point clouds using Open3D.

Usage:
    python visualize_masked_pointclouds.py --scratch_dir /path/to/scratch_dir [--timestamp <timestamp_left-timestamp_right>]

Requirements:
    pip install open3d numpy

File organization:
    pointcloud: {scratch_dir}/disparity_outputs/{timestamp}/cloud_denoise.ply
    masks:       {scratch_dir}/masks/{timestamp}/mask_{idx}_area{mask_area}.npy

Assumptions:
    - Each mask .npy is either:
        * a boolean array of length N (number of points) where True indicates keep
        * an integer array of indices of points to keep
        * a float array of length N with values in [0,1], thresholded at 0.5
    - If mask array length matches number of points in the loaded point cloud, treat as boolean/float mask.
    - Otherwise, if dtype is integer, treat as indices.
"""

import open3d as o3d
import numpy as np
import os
import glob
import argparse
from pathlib import Path
import copy
import matplotlib.pyplot as plt

def load_pointcloud(ply_path):
    """
    Load a point cloud from a .ply file using Open3D.
    """
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"Point cloud file not found: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        print(f"Warning: loaded point cloud is empty: {ply_path}")
    return pcd

def load_mask(mask_path, num_points=None):
    """
    Load a mask from a .npy file and convert to a boolean mask over point cloud indices.
    Parameters:
        - mask_path: path to .npy file
        - num_points: integer, number of points in the point cloud
    Returns:
        - mask: 1D boolean numpy array of length num_points, True for points to keep
    """
    arr = np.load(mask_path)
        # plot the mask
    plt.figure(figsize=(8, 6))
    plt.imshow(arr, cmap='gray', aspect='auto')
    plt.title(f'Mask Visualization: ')
    plt.colorbar(label='Mask Value')
    plt.xlabel('Point Index')
    plt.ylabel('Mask Value')
    plt.show()

    # Boolean mask of same length
    if num_points is not None and arr.dtype == bool and arr.shape[0] == num_points:
        mask = arr
    # Float mask [0,1] of same length
    elif num_points is not None and arr.dtype.kind == 'f' and arr.shape[0] == num_points:
        # Threshold at 0.5
        mask = arr > 0.5
    # Integer indices
    elif arr.dtype.kind in ('i', 'u'):
        if num_points is None:
            raise ValueError(f"num_points must be provided for index mask: {mask_path}")
        mask = np.zeros(num_points, dtype=bool)
        idx = arr
        # Filter valid indices
        idx_valid = idx[(idx >= 0) & (idx < num_points)]
        mask[idx_valid] = True
    else:
        raise ValueError(f"Mask array shape/dtype not recognized or mismatch with pointcloud: {mask_path}")
    return mask

def visualize_pointclouds(pcd, masked_pcds, mask_names):
    """
    Visualize the original point cloud and masked point clouds.
    - Original is colored gray.
    - Masked points are colored red and overlaid on gray background.
    """
    # Prepare original in light gray
    pcd_orig = copy.deepcopy(pcd)
    num_pts = np.asarray(pcd_orig.points).shape[0]
    gray = np.array([0.7, 0.7, 0.7], dtype=np.float64)
    colors = np.tile(gray, (num_pts, 1))
    pcd_orig.colors = o3d.utility.Vector3dVector(colors)

    # Display the original
    print("Displaying original point cloud. Close the window to proceed to masked views.")
    o3d.visualization.draw_geometries([pcd_orig], window_name='Original Point Cloud')

    # For each masked point cloud, overlay in red
    for mp, name in zip(masked_pcds, mask_names):
        # Color masked points red
        mp_colored = copy.deepcopy(mp)
        num_mp = np.asarray(mp_colored.points).shape[0]
        red = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        colors_m = np.tile(red, (num_mp, 1))
        mp_colored.colors = o3d.utility.Vector3dVector(colors_m)

        # Combine original gray and masked red
        combined = pcd_orig + mp_colored
        print(f"Displaying masked view for mask: {name}. Close the window to continue.")
        o3d.visualization.draw_geometries([combined], window_name=f'Masked: {name}')

def process_timestamp(scratch_dir, timestamp):
    """
    Load the point cloud and masks for a given timestamp, apply masks, and visualize.
    """
    disp_dir = os.path.join(scratch_dir, 'disparity_outputs', timestamp)
    pcd_path = os.path.join(disp_dir, 'cloud_denoise.ply')
    try:
        pcd = load_pointcloud(pcd_path)
    except Exception as e:
        print(f"Error loading point cloud for timestamp {timestamp}: {e}")
        return
    num_points = np.asarray(pcd.points).shape[0]
    print(f"Loaded point cloud for timestamp {timestamp}, number of points: {num_points}")

    mask_dir = os.path.join(scratch_dir, 'masks', timestamp)
    if not os.path.isdir(mask_dir):
        print(f"Mask directory not found for timestamp {timestamp}: {mask_dir}")
        return
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, 'mask_*.npy')))
    if not mask_paths:
        print(f"No mask files found in {mask_dir}")
        return

    masked_pcds = []
    mask_names = []
    print(f"pcl shape as array {np.asarray(pcd.points).shape}")
    print(f"pcl as array {np.asarray(pcd.points)}")
    # extract only x and y coordinates
    flattened_pcd = np.asarray(pcd.points)[:, :2]
    
    print(f"flattened pcl as array {flattened_pcd}")
    for mpath in mask_paths:
        try:
            mask = load_mask(mpath, num_points=num_points)
        except Exception as e:
            print(f"Skipping mask {mpath}: {e}")
            continue
        print(f"Loaded mask from {mpath}, shape: {mask.shape}, mask: {mask}")

        exit()
        indices = np.where(mask)[0].tolist()
        if not indices:
            print(f"Mask {mpath} is empty after processing. Skipping.")
            continue
        mp = pcd.select_by_index(indices)
        masked_pcds.append(mp)
        mask_name = os.path.basename(mpath)
        mask_names.append(mask_name)

    if not masked_pcds:
        print(f"No valid masks to visualize for timestamp {timestamp}.")
        return

    visualize_pointclouds(pcd, masked_pcds, mask_names)

def main():
    parser = argparse.ArgumentParser(description="Visualize point clouds with and without masks using Open3D.")
    
    parser.add_argument('--timestamp', type=str, default=None,
                        help='Specific timestamp directory to process (format: timestamp_left-timestamp_right). '
                             'If omitted, process all subdirectories in disparity_outputs.')
    args = parser.parse_args()

    scratch_dir = Path("/home/armjetson/nvme/data")

    if args.timestamp:
        print(f"Processing timestamp: {args.timestamp}")
        process_timestamp(scratch_dir, args.timestamp)
    else:
        # Process all subdirectories under disparity_outputs
        disp_base = os.path.join(scratch_dir, 'disparity_outputs')
        if not os.path.isdir(disp_base):
            print(f"Disparity outputs directory not found: {disp_base}")
            return
        subdirs = [d for d in os.listdir(disp_base) if os.path.isdir(os.path.join(disp_base, d))]
        if not subdirs:
            print(f"No timestamp subdirectories found in {disp_base}.")
            return
        for ts in sorted(subdirs):
            print(f"\n=== Processing timestamp: {ts} ===")
            process_timestamp(scratch_dir, ts)

if __name__ == '__main__':
    main()
