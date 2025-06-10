#!/usr/bin/python3
import os
import small_gicp
import open3d
import numpy
import argparse

def load_and_filter(filepath, z_thresh=1.5):
    """
    Load a .ply point cloud, then discard any points with z ≥ z_thresh.
    """
    pcd = open3d.io.read_point_cloud(filepath)
    pts = numpy.asarray(pcd.points)
    mask = pts[:, 2] < z_thresh
    if not mask.all():
        pcd = pcd.select_by_index(numpy.where(mask)[0])
    return pcd

def preprocess_for_gicp(pcd, voxel_size=1.0):
    """
    Take an Open3D point cloud, convert to numpy, then downsample via small_gicp.preprocess_points.
    Returns (downsampled_points, kdtree).
    """
    pts = numpy.asarray(pcd.points)
    down, tree = small_gicp.preprocess_points(pts, downsampling_resolution=voxel_size)
    return down, tree
import open3d as o3d

def visualize_with_saved_camera(geometry_list, camera_json_path):
    """
    Opens an Open3D window, adds all geometries in geometry_list,
    applies the saved camera from camera_json_path, and then runs.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Pairwise Alignment (with preset camera)",
        width=1280,
        height=720
    )

    # Add all your geometries (pointclouds, meshes, etc.)
    for geom in geometry_list:
        vis.add_geometry(geom)

    # Load the saved camera parameters (PinholeCameraParameters format)
    params = o3d.io.read_pinhole_camera_parameters(camera_json_path)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    vis.run()
    vis.destroy_window()

def align_and_visualize(target_fp, source_fp, z_thresh=1.5, voxel_size=1.0):
    """
    1) Load target and source, filter by z < z_thresh
    2) Preprocess (voxel downsample + tree)
    3) Run small_gicp.align
    4) Print fitness/RMSE (if available) and T_target_source
    5) Transform source, paint:
       - target → green, source→ blue, plus coordinate frame origin in red
    6) draw_geometries([...]) and wait until window is closed
    """
    # --- Load & filter ---
    target_raw = load_and_filter(target_fp, z_thresh=z_thresh)
    source_raw = load_and_filter(source_fp, z_thresh=z_thresh)

    # Paint them distinct colors:
    target_o3d = target_raw
    source_o3d = source_raw

    # --- Preprocess for GICP ---
    target_np, target_tree = preprocess_for_gicp(target_o3d, voxel_size=voxel_size)
    source_np, source_tree = preprocess_for_gicp(source_o3d, voxel_size=voxel_size)

    # --- Align ---
    print(f"\n--- Aligning ---\n  target: {os.path.basename(target_fp)}\n  source: {os.path.basename(source_fp)}")
    result = small_gicp.align(target_np, source_np, target_tree)
    # Try to print fitness / RMSE if available
    if hasattr(result, "fitness"):
        print(f"  fitness: {result.fitness:.4f}")
    if hasattr(result, "inlier_rmse"):
        print(f"  inlier_rmse: {result.inlier_rmse:.4f}")
    print("  T_target_source:\n", result.T_target_source)

    # --- Transform source into target frame ---
    source_o3d.transform(result.T_target_source)

    # --- Coordinate frame marker at origin ---
    origin_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0.0, 0.0, 0.0]
    ).paint_uniform_color([1.0, 0.0, 0.0])

    # --- Visualize ---
    # open3d.visualization.draw_geometries(
    #     [target_o3d, source_o3d, origin_frame],
    #     window_name="Pairwise Alignment Debug",
    #     width=1280,
    #     height=720
    # )
    visualize_with_saved_camera(
        [target_o3d, source_o3d, origin_frame],
        camera_json_path="ScreenCamera_2025-06-06-12-36-47.json"  # Adjust path as needed
    )

def main():
    parser = argparse.ArgumentParser(
        description="Step through consecutive PLY pairs and visualize their GICP alignment."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/recording",
        help="Folder containing ordered .ply frames"
    )
    parser.add_argument(
        "--z_thresh",
        type=float,
        default=1.5,
        help="Discard points with z ≥ this value before alignment"
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=1.0,
        help="Downsampling resolution passed to small_gicp.preprocess_points"
    )
    args = parser.parse_args()

    # --- Gather & sort .ply files ---
    all_files = sorted([
        os.path.join(args.dataset_path, f)
        for f in os.listdir(args.dataset_path)
        if f.lower().endswith(".ply")
    ])
    if len(all_files) < 2:
        print(f"Need at least 2 .ply files in '{args.dataset_path}'.")
        return

    # Step through consecutive pairs
    for i in range(0,len(all_files) - 1,5):
        target_fp = all_files[i]
        source_fp = all_files[i + 5]

        align_and_visualize(
            target_fp,
            source_fp,
            z_thresh=args.z_thresh,
            voxel_size=args.voxel_size

        )
        # After the user closes the window, the loop continues to the next pair.

    print("\nDone stepping through all pairs.")

if __name__ == "__main__":
    main()
