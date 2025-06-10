import small_gicp
import open3d
import numpy

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

# Dummy bounds (tune these later)
XMIN, XMAX = -0.5, 0.6
YMIN, YMAX = -1.0, 1.0
ZMIN, ZMAX = -0.3, 1.0

target_filepath = 'data/recording/frame_0110.ply'
source_filepath = 'data/recording/frame_0120.ply'

print(f"Target point cloud: {target_filepath}")
print(f"Source point cloud: {source_filepath}")

color = False
if color:
    target_o3d = open3d.io.read_point_cloud(target_filepath).paint_uniform_color([0, 0.5, 0])
    source_o3d = open3d.io.read_point_cloud(source_filepath).paint_uniform_color([0, 0, 0.5])
else:
    target_o3d = open3d.io.read_point_cloud(target_filepath)
    source_o3d = open3d.io.read_point_cloud(source_filepath)

# print(f"Target point cloud: {numpy.asarray(target_o3d.points)}")
# exit()
# # process point clouds to remove the points that are too far away from the origin
distance_threshold = 1.7

# print(f"Number of points in target point cloud before filtering: {len(target_o3d.points)}")
# print(f"Number of points in source point cloud before filtering: {len(source_o3d.points)}")

# target_o3d = target_o3d.select_by_index(numpy.where(numpy.asarray(target_o3d.points)[:, 2] < distance_threshold)[0])
# source_o3d = source_o3d.select_by_index(numpy.where(numpy.asarray(source_o3d.points)[:, 2] < distance_threshold)[0])

# print(f"Number of points in target point cloud after filtering: {len(target_o3d.points)}")
# print(f"Number of points in source point cloud after filtering: {len(source_o3d.points)}")

# Crop the point clouds to a box defined by the bounds
target_o3d = crop_box(target_o3d, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
source_o3d = crop_box(source_o3d, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)


max_val_target = numpy.max(numpy.asarray(target_o3d.points))
max_val_source = numpy.max(numpy.asarray(source_o3d.points))

min_val_target = numpy.min(numpy.asarray(target_o3d.points))
min_val_source = numpy.min(numpy.asarray(source_o3d.points))

target, target_tree = small_gicp.preprocess_points(numpy.asarray(target_o3d.points), downsampling_resolution=1.0)
# print(f"Target tree: {target_tree}")
source, source_tree = small_gicp.preprocess_points(numpy.asarray(source_o3d.points), downsampling_resolution=1.0)
# print(f"Source tree: {source_tree}")
result = small_gicp.align(target, source, target_tree)
# print(f"Result: {result}")
print(f"Transformation matrix:\n{result.T_target_source}")

# create max and min values from the point clouds to visualize the point clouds

origin = numpy.array([0, 0, 0])
origin = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=origin).paint_uniform_color([1, 0, 0])

point1 = numpy.array([0.0, 0.0,max_val_target])
# origon = open3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=20).translate(point).paint_uniform_color([1, 0, 0])
point2 = numpy.array([min_val_target, 0.0, 0.0])
# origon = open3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=20).translate(point).paint_uniform_color([1, 0, 0])
point3 = numpy.array([0.0, min_val_target, 0.0])
# origon = open3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=20).translate(point).paint_uniform_color([1, 0, 0])

pointx = numpy.array([0.0, 0.0, distance_threshold])

point1 = open3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=20).translate(point1).paint_uniform_color([1, 0, 0])
point2 = open3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=20).translate(point2).paint_uniform_color([1, 0, 0])
point3 = open3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=20).translate(point3).paint_uniform_color([1, 0, 0])
pointx = open3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=20).translate(pointx).paint_uniform_color([1, 0, 0])

# Create an AxisAlignedBoundingBox from the same bounds used for cropping
min_bound = numpy.array([XMIN, YMIN, ZMIN])
max_bound = numpy.array([XMAX, YMAX, ZMAX])
bbox = open3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
bbox.color = (1.0, 0.0, 0.0)  # red box

# After cropping and alignment, for quick check you can visualize:
# e.g., before alignment:
# open3d.visualization.draw_geometries([target_o3d, bbox], window_name="Target with Crop Box")
# open3d.visualization.draw_geometries([source_o3d, bbox], window_name="Source with Crop Box")

# Or together:
open3d.visualization.draw_geometries([target_o3d, source_o3d, bbox], window_name="Both with Crop Box")



source_o3d.transform(result.T_target_source)
open3d.visualization.draw_geometries([target_o3d, source_o3d, origin, pointx])

#TODO: Try to visualize a series of point clouds to see when the alignment fails