#!/usr/bin/python3
import open3d

def main():
    # Simply load the saved map and display it
    path = "/home/armjetson/nvme/test_outputs/cloud_denoise.ply"
    pcd = open3d.io.read_point_cloud(path)
    if pcd.is_empty():
        print("ERROR: 'map.ply' is empty or not found.")
        return

    # You can set a uniform color, or comment out to show per‐vertex colors (if any)
    # pcd.paint_uniform_color([0.8, 0.3, 0.1])  # a reddish/orange hue

    open3d.visualization.draw_geometries(
        [pcd],
        window_name="Accumulated Map",
        width=1280,
        height=720,
        left=50,
        top=50,
        point_show_normal=False
    )

if __name__ == "__main__":
    main()
