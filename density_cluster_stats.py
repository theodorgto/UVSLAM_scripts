#!/usr/bin/python3
import os
import argparse
import numpy as np
import open3d as o3d

def load_and_filter(filepath, z_thresh):
    """
    Load a .ply point cloud and remove points with z ≥ z_thresh.
    Returns an Open3D point cloud.
    """
    pcd = o3d.io.read_point_cloud(filepath)
    pts = np.asarray(pcd.points)
    if z_thresh is not None:
        mask = pts[:, 2] < z_thresh
        if not mask.all():
            pcd = pcd.select_by_index(np.where(mask)[0])
    return pcd

def compute_dbscan_clusters(pcd, eps, min_points):
    """
    Run DBSCAN clustering on pcd.
    Returns a numpy array of labels (length = number of points).
    Label = -1 means "noise"; otherwise labels = 0,1,2,...
    """
    # The argument `print_progress=False` simply silences console spam on large clouds
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    return labels

def main():
    parser = argparse.ArgumentParser(
        description="For each PLY: filter by z, run DBSCAN, count largest cluster."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/recording",
        help="Folder containing .ply frames"
    )
    parser.add_argument(
        "--z_thresh",
        type=float,
        default=1.7,
        help="Discard points with z ≥ this value (set to None to disable)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.05,
        help="DBSCAN eps parameter (meters)"
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=10,
        help="Minimum number of points to form a cluster"
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

    print(
        "Filename, N_total, N_largest_cluster, Percent_largest (%)"
    )
    files = files[:100]  # Limit to first 100 files for performance
    for fp in files:
        pcd = load_and_filter(fp, z_thresh=args.z_thresh)
        N_total = len(pcd.points)
        if N_total == 0:
            print(f"{os.path.basename(fp)}, 0, 0, 0.0")
            continue

        # DBSCAN clustering
        labels = compute_dbscan_clusters(pcd, eps=args.eps, min_points=args.min_points)
        # Count points per label (ignore noise = label -1)
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if counts.size == 0:
            # No valid clusters found
            N_largest = 0
        else:
            N_largest = counts.max()

        pct = 100.0 * N_largest / N_total
        print(f"{os.path.basename(fp)}, {N_total}, {N_largest}, {pct:.2f}")

if __name__ == "__main__":
    main()
