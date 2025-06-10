#!/usr/bin/python3
import os
import argparse
import numpy as np
import open3d as o3d
import small_gicp

def load_point_cloud(filepath):
    """
    Load a .ply point cloud (no distance‐threshold filtering).
    """
    return o3d.io.read_point_cloud(filepath)

def segment_plane(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    """
    Run RANSAC plane segmentation on the point cloud `pcd`.
    Returns: (plane_model, inlier_indices)
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    return plane_model, inliers

def preprocess_for_gicp(pcd, voxel_size=1.0):
    """
    Take an Open3D point cloud, convert to numpy, then downsample via small_gicp.preprocess_points.
    Returns (downsampled_points, kdtree).
    """
    pts = np.asarray(pcd.points)
    down, tree = small_gicp.preprocess_points(pts, downsampling_resolution=voxel_size)
    return down, tree

def visualize_with_saved_camera(geometry_list, camera_json_path):
    """
    Opens an Open3D window, adds all geometries in geometry_list,
    applies the saved camera from camera_json_path, and then runs.
    """
    params = o3d.io.read_pinhole_camera_parameters(camera_json_path)
    intr = params.intrinsic
    w, h = intr.width, intr.height

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Pairwise Alignment (no plane)",
        width=w,
        height=h
    )

    for geom in geometry_list:
        vis.add_geometry(geom)

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    vis.run()
    vis.destroy_window()
def crop_box(pcd, xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Keep only points within the axis-aligned box [xmin,xmax]×[ymin,ymax]×[zmin,zmax].
    """

    pts = np.asarray(pcd.points)
    mask = (
        (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
        (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
        (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
    )
    indices = np.where(mask)[0]
    return pcd.select_by_index(indices)



def align_and_evaluate(target_fp, source_fp, voxel_size,
                       plane_dist, plane_ransac_n, plane_iter, camera_json_path):
    """
    1) Load target and source (no filtering)
    2) Segment & remove dominant plane from each
    3) Run GICP on (a) full clouds, (b) plane‐removed clouds
    4) Print metrics for both
    5) Visualize alignment of plane‐removed clouds
    """
    # Load both point clouds (no z filtering)
    target_raw = load_point_cloud(target_fp)
    source_raw = load_point_cloud(source_fp)

    # filter out points with z > 1.7 (optional, can be adjusted)
    # target_raw = target_raw.select_by_index(
    #     np.where(np.asarray(target_raw.points)[:, 2] < 1.7)[0]
    # )
    # source_raw = source_raw.select_by_index(
    #     np.where(np.asarray(source_raw.points)[:, 2] < 1.7)[0]
    # )



    # Crop the point clouds to a box defined by the bounds
    # bounds
    XMIN, XMAX = -0.4, 0.6
    YMIN, YMAX = -1.0, 1.0
    ZMIN, ZMAX = -0.3, 1.3

    target_raw = crop_box(target_raw, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
    source_raw = crop_box(source_raw, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

    print(f"Target point cloud size: {len(target_raw.points)}")
    print(f"Source point cloud size: {len(source_raw.points)}")

    # Segment plane on target
    plane_model_t, inliers_t = segment_plane(
        target_raw,
        distance_threshold=plane_dist,
        ransac_n=plane_ransac_n,
        num_iterations=plane_iter
    )
    Ntot_t = len(target_raw.points)
    Nplane_t = len(inliers_t)
    pct_t = 100.0 * Nplane_t / Ntot_t if Ntot_t > 0 else 0.0
    target_noplane = target_raw.select_by_index(inliers_t, invert=True)

    # Segment plane on source
    plane_model_s, inliers_s = segment_plane(
        source_raw,
        distance_threshold=plane_dist,
        ransac_n=plane_ransac_n,
        num_iterations=plane_iter
    )
    Ntot_s = len(source_raw.points)
    Nplane_s = len(inliers_s)
    pct_s = 100.0 * Nplane_s / Ntot_s if Ntot_s > 0 else 0.0
    source_noplane = source_raw.select_by_index(inliers_s, invert=True)

    print(f"\n--- Pair: {os.path.basename(target_fp)} ↔ {os.path.basename(source_fp)} ---")
    print(f"Target total pts: {Ntot_t}, plane inliers: {Nplane_t} ({pct_t:.1f}%)")
    print(f"Source total pts: {Ntot_s}, plane inliers: {Nplane_s} ({pct_s:.1f}%)")

    # ------- GICP on full clouds -------
    # tgt_full_down, tgt_full_tree = preprocess_for_gicp(target_raw, voxel_size)
    # src_full_down, src_full_tree = preprocess_for_gicp(source_raw, voxel_size)
    # result_full = small_gicp.align(tgt_full_down, src_full_down, tgt_full_tree)
    
    print(f"taget_raw points: {len(target_raw.points)}")
    print(f"source_raw points: {len(source_raw.points)}")

    # 1) Convert Open3D clouds to numpy arrays
    tgt_pts = np.asarray(target_raw.points)     # raw or already‐filtered cloud
    src_pts = np.asarray(source_raw.points)

    print(f"Target points shape: {tgt_pts[:,0].shape}")



    # 2) Call the “array” overload of align(...), specifying VGICP
    result_full = small_gicp.align(
        tgt_pts,                            # target Nx3 float64
        src_pts,                            # source Nx3 float64
        init_T_target_source=np.eye(4),     # usually identity if no prior
        registration_type='VGICP',          # switch to VGICP
        voxel_resolution=1.0 / 1000,               # size of Gaussian voxels (meters)
        downsampling_resolution=voxel_size / 1000, # your previous downsample rate
        max_correspondence_distance=1.0,    # or a smaller value if you like
        num_threads=1,                      # or whatever you were using
        max_iterations=20,                  # same as before or tuned
        verbose=False
    )

    print(f"transform:\n{result_full.T_target_source}")


    print(f"Full‐cloud VGICP error: {result_full.error:.4f}")
    # print(f"  T_target_source:\n{result_full}")

    # fitness_full = getattr(result_full, "fitness", None)
    # rmse_full = getattr(result_full, "inlier_rmse", None)
    # print("Full‐cloud GICP:")
    # if fitness_full is not None:
    #     print(f"  fitness:    {fitness_full:.4f}")
    # if rmse_full is not None:
    #     print(f"  inlier_rmse:{rmse_full:.4f}")
    # print(f"  transform:\n{result_full.T_target_source}")

    # ------- GICP on plane‐removed clouds -------
    # tgt_np_down, tgt_np_tree = preprocess_for_gicp(target_noplane, voxel_size)
    # src_np_down, src_np_tree = preprocess_for_gicp(source_noplane, voxel_size)
    # result_np = small_gicp.align(tgt_np_down, src_np_down, tgt_np_tree)

    tgt_np_pts = np.asarray(target_noplane.points)
    src_np_pts = np.asarray(source_noplane.points)

    result_np = small_gicp.align(
        tgt_np_pts,
        src_np_pts,
        init_T_target_source=np.eye(4),
        registration_type='VGICP',
        voxel_resolution=1.0 / 1000,
        downsampling_resolution=voxel_size / 1000,
        max_correspondence_distance=1.0,
        num_threads=1,
        max_iterations=20,
        verbose=False
    )


    print(f"Plane‐removed VGICP error: {result_np.error:.4f}")
    # print(f"  T_target_source:\n{result_np}")

    # fitness_np = getattr(result_np, "fitness", None)
    # rmse_np = getattr(result_np, "inlier_rmse", None)
    # print("No‐plane GICP:")
    # if fitness_np is not None:
    #     print(f"  fitness:    {fitness_np:.4f}")
    # if rmse_np is not None:
    #     print(f"  inlier_rmse:{rmse_np:.4f}")
    # print(f"  transform:\n{result_np.T_target_source}")

    # ------- Visualize alignment of plane‐removed clouds -------
    target_vis = o3d.geometry.PointCloud(target_noplane)#.paint_uniform_color([0.0, 0.5, 0.0])
    source_vis = o3d.geometry.PointCloud(source_noplane)#.paint_uniform_color([0.0, 0.0, 0.5])
    source_vis.transform(result_np.T_target_source)

    min_bound = np.array([XMIN, YMIN, ZMIN])
    max_bound = np.array([XMAX, YMAX, ZMAX])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    bbox.color = (1.0, 0.0, 0.0)

    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    ).paint_uniform_color([1.0, 0.0, 0.0])

    visualize_with_saved_camera(
        [target_vis, source_vis, bbox, origin_frame],
        camera_json_path
    )

    # # -------- Visualize original clouds with planes --------
    # target_vis = o3d.geometry.PointCloud(target_raw)
    # source_vis = o3d.geometry.PointCloud(source_raw)
    # source_vis.transform(result_full.T_target_source)

    # target_plane = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=0.5, origin=[0, 0, 0]
    # ).paint_uniform_color([1.0, 0.0, 0.0])

    # visualize_with_saved_camera(
    #     [target_vis, source_vis, target_plane],
    #     camera_json_path
    # )


def main():
    parser = argparse.ArgumentParser(
        description="Step through PLY pairs, remove dominant planes, and compare GICP metrics."
    )
    parser.add_argument(
        "--dataset_path", type=str, default="data/recording",
        help="Folder containing ordered .ply frames"
    )
    parser.add_argument(
        "--voxel_size", type=float, default=1.0,
        help="Downsampling resolution for small_gicp.preprocess_points"
    )
    parser.add_argument(
        "--plane_dist", type=float, default=0.02,
        help="RANSAC distance threshold for plane inliers (meters)"
    )
    parser.add_argument(
        "--plane_ransac_n", type=int, default=3,
        help="Number of points sampled to fit each plane hypothesis"
    )
    parser.add_argument(
        "--plane_iter", type=int, default=1000,
        help="Number of RANSAC iterations for plane segmentation"
    )
    parser.add_argument(
        "--step", type=int, default=5,
        help="Step size between frames (e.g. 1 to compare consecutive, 5 to skip every 5)"
    )
    parser.add_argument(
        "--view_json", type=str,
        default="ScreenCamera_2025-06-06-12-36-47.json",
        help="Path to saved PinholeCameraParameters JSON for visualization"
    )
    args = parser.parse_args()

    # Gather & sort all .ply files
    all_files = sorted([
        os.path.join(args.dataset_path, f)
        for f in os.listdir(args.dataset_path)
        if f.lower().endswith(".ply")
    ])
    if len(all_files) < 2:
        print(f"Need at least 2 .ply files in '{args.dataset_path}'.")
        return

    all_files = all_files[100:]  # Skip first 100 files

    # Step through pairs
    for i in range(0, len(all_files) - args.step, args.step):
        tgt_fp = all_files[i]
        src_fp = all_files[i + args.step]
        align_and_evaluate(
            tgt_fp, src_fp,
            voxel_size=args.voxel_size,
            plane_dist=args.plane_dist,
            plane_ransac_n=args.plane_ransac_n,
            plane_iter=args.plane_iter,
            camera_json_path=args.view_json
        )

    print("\nDone stepping through all pairs.")

if __name__ == "__main__":
    main()
