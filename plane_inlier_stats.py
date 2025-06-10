#!/usr/bin/python3
import os
import argparse
import numpy as np
import open3d as o3d

def load_point_cloud(filepath):
    """
    Load a .ply point cloud (no filtering).
    """
    pcd = o3d.io.read_point_cloud(filepath)
    return pcd

def compute_plane_inliers(pcd, distance_threshold, ransac_n, num_iterations):
    """
    Run RANSAC plane segmentation on the point cloud `pcd`.
    Returns:
      - plane_model: coefficients [a,b,c,d] of the best‐fit plane ax+by+cz+d=0
      - inlier_indices: list of indices of pcd.points that lie within distance_threshold of that plane
    """
    plane_model, inlier_indices = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    return plane_model, inlier_indices

def visualize_plane(pcd, inlier_indices):
    """
    Given an Open3D point cloud `pcd` and a list of `inlier_indices`,
    paint plane‐inliers red and everything else gray, then show the result
    alongside a coordinate frame at the origin.
    """
    # Make a copy so we don't overwrite the original
    colored = o3d.geometry.PointCloud(pcd)
    N = len(colored.points)
    colors = np.tile([0.5, 0.5, 0.5], (N, 1))  # default gray

    # Paint inliers red
    colors[inlier_indices, :] = np.array([1.0, 0.0, 0.0])
    colored.colors = o3d.utility.Vector3dVector(colors)

    # Add a little coordinate frame at (0,0,0)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )

    o3d.visualization.draw_geometries(
        [colored, origin_frame],
        window_name="Plane Inliers (red) vs. Rest (gray)",
        width=800,
        height=600
    )

def main():
    parser = argparse.ArgumentParser(
        description="For each PLY: run plane‐RANSAC (no filtering), print stats, and visualize inliers."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/recording",
        help="Folder containing .ply frames"
    )
    parser.add_argument(
        "--ransac_dist",
        type=float,
        default=0.02,
        help="RANSAC distance threshold (in meters) for plane inliers"
    )
    parser.add_argument(
        "--ransac_n",
        type=int,
        default=3,
        help="Number of points sampled to fit each plane hypothesis"
    )
    parser.add_argument(
        "--ransac_iter",
        type=int,
        default=1000,
        help="Number of RANSAC iterations"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, pop up a window showing plane inliers (red) vs rest (gray) for each frame"
    )
    args = parser.parse_args()

    # Gather sorted list of .ply files
    files = sorted([
        os.path.join(args.dataset_path, f)
        for f in os.listdir(args.dataset_path)
        if f.lower().endswith(".ply")
    ])
    if not files:
        print(f"No .ply files found in '{args.dataset_path}'")
        return

    print("Filename, N_total, N_plane_inliers, Percent_plane (%)")
    files = files[:100]  # Limit to first 100 files for performance

    avg_percent_plane = 0.0

    for fp in files:
        pcd = load_point_cloud(fp)
        N_total = len(pcd.points)
        if N_total == 0:
            print(f"{os.path.basename(fp)}, 0, 0, 0.00")
            continue

        # Run RANSAC plane segmentation on the full cloud
        plane_model, inliers = compute_plane_inliers(
            pcd,
            distance_threshold=args.ransac_dist,
            ransac_n=args.ransac_n,
            num_iterations=args.ransac_iter
        )
        N_plane = len(inliers)
        pct = 100.0 * N_plane / N_total

        print(f"{os.path.basename(fp)}, {N_total}, {N_plane}, {pct:.2f}")

        avg_percent_plane += pct / len(files)

        # If requested, visualize the inliers vs. the rest
        if args.visualize:
            visualize_plane(pcd, inliers)
    
    print(f"\nAverage percent of plane inliers across all frames: {avg_percent_plane:.2f}%")

if __name__ == "__main__":
    main()
