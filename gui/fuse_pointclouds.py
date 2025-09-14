import numpy as np
import os
import sys

import re
import json
import argparse
import cv2
import open3d as o3d
import shutil
from PIL import Image
import tqdm



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter_path",
                        default="/home/user/PycharmProjects/animal_pose_-gui/example_operator1/experiment/stereo_parameter.json",
                        # required=True,
                        help='path of json file storing the cameras parameters', )

    parser.add_argument("--base_path",
                        default="/home/user/PycharmProjects/animal_pose_-gui/example_operator1/experiment/test",
                        help="path contains multiview images")

    parser.add_argument("--num_devices",
                        default=4,
                        type=int,
                        help="number of devies", )

    parser.add_argument('--z_far',
                        default=10.0, type=float,
                        help='max depth to clip in point cloud')

    parser.add_argument("--mask_mix_ratio", type=float,
                        default=0.0,
                        help="ratio of mask mixing with RGB")

    parser.add_argument("--object_num", type=int,
                        default=3,
                        help="number of object in the frame", )

    args = parser.parse_args()
    return args

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def depth2pts_np(depth_map, cam_intrinsic, cam_extrinsic=np.eye(4)):
    feature_grid = get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

    uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
    cam_points = uv * np.reshape(depth_map, (1, -1))

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    world_points = np.matmul(R_inv, cam_points - t).transpose()
    return world_points


def get_pixel_grids_np(height, width):
    x_linspace = np.linspace(0.5, width - 0.5, width)
    y_linspace = np.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = np.reshape(x_coordinates, (1, -1))
    y_coordinates = np.reshape(y_coordinates, (1, -1))
    ones = np.ones_like(x_coordinates).astype(float)
    grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)
    return grid


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def process_pointcloud(pointcloud: o3d.geometry.PointCloud):
                
    voxel_size = 0.0005  
    downsampled_pcd = pointcloud.voxel_down_sample(voxel_size=voxel_size)

    inlier_cloud = downsampled_pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=2)[0]
    inlier_cloud = inlier_cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=2)[0]
    inlier_cloud = inlier_cloud.remove_statistical_outlier(nb_neighbors=50, std_ratio=1)[0]

    return inlier_cloud

    

def main():
  
    # Initialization
    args = parse_args()

    base_path = args.base_path
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path {base_path} does not exist.")

    device_parameters = json.load(open(args.parameter_path, 'r', encoding='utf-8-sig'))

    output_path = os.path.join(base_path, 'pointclouds')
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    pointclouds = {}
    frame_keys = []

    for device_idx in range(args.num_devices):
        device = f"device{device_idx}"
        device_path = os.path.join(base_path, device)
        if not os.path.exists(device_path):
            raise FileNotFoundError(f"Device path {device_path} does not exist.")
        device_depth_dir = os.path.join(device_path, 'depth')
        device_rgb_dir = os.path.join(device_path, 'left_camera_rectified')
        device_mask_dir = os.path.join(device_path, 'mask')

        device_frames = sorted(
            [img for img in os.listdir(device_depth_dir) if img.lower().endswith(('png'))],
            key=natural_sort_key)
        device_frames = [os.path.splitext(frame)[0] for frame in device_frames]
        if device_idx == 0:
            frame_keys = device_frames
        else:
            if frame_keys != device_frames:
                raise ValueError(f"Frame keys mismatch between devices: {frame_keys} and {device_frames}")

        cam_intrinsic = np.array(device_parameters[device]["stereo_params"]["intrinsic"])
        cam_extrinsic = np.linalg.inv(np.array(device_parameters[device]['pose_wrt_world']))

        for frame in frame_keys:
            if frame not in pointclouds:
                pointclouds[frame] = {}

            depth = cv2.imread(os.path.join(device_depth_dir, f"{frame}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            mask, _ = load_ann_png(os.path.join(device_mask_dir, f"{frame}.png"))

            # remove points with depth greater than z_far
            depth[depth > args.z_far] = 0
            mask[depth == 0] = 0

            rgb = cv2.imread(os.path.join(device_rgb_dir, f"{frame}.jpg"), cv2.IMREAD_UNCHANGED)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            mask_rgb = cv2.imread(os.path.join(device_mask_dir, f"{frame}.png"), cv2.IMREAD_UNCHANGED)
            mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if args.mask_mix_ratio > 0:
                rgb = rgb * (1 - args.mask_mix_ratio) + mask_rgb * args.mask_mix_ratio

            for obj_idx in range(1, args.object_num + 1):
                if f"obj{obj_idx}" not in pointclouds[frame]:
                    pointclouds[frame][f"obj{obj_idx}"] = {}
                try:
                    object_mask = get_per_obj_mask(mask)[obj_idx]
                    object_depth = np.zeros_like(depth)
                    object_depth[object_mask] = depth[object_mask]

                except KeyError:
                    # print(f"Warning: video {video} device {device_number} frame {filename}, Object {obj_idx} does not have a mask. Skipping.")
                    object_depth = np.zeros_like(depth)

                pts = depth2pts_np(object_depth, cam_intrinsic, cam_extrinsic)
                object_mask = object_mask.reshape(-1)
                pts = pts[object_mask]
                rgb_values = rgb.reshape(-1, 3)[object_mask]
                pointclouds[frame][f"obj{obj_idx}"][device] = {
                    "points": pts,
                    "colors": rgb_values
                }
            
    # Process point clouds
    for frame, frame_data in pointclouds.items():
        print(f"Processing frame {frame}...")
        frame_path = os.path.join(output_path, frame)
        os.makedirs(frame_path, exist_ok=True)
        combined_points = []
        combined_colors = []
        for obj_name, obj_data in frame_data.items():
            obj_points = []
            obj_colors = []
            for device, data in obj_data.items():
                # print(f"object: {obj_name}, Device: {device}, Points: {data['points'].shape[0]}")
                obj_points.append(data['points'])
                obj_colors.append(data['colors'])
            obj_points = np.concatenate(obj_points, axis=0)
            obj_colors = np.concatenate(obj_colors, axis=0)
            print(f"Object {obj_name} has {obj_points.shape[0]} points.")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points)
            pcd.colors = o3d.utility.Vector3dVector(obj_colors)
            pcd = process_pointcloud(pcd)
            o3d.io.write_point_cloud(os.path.join(frame_path, f"{obj_name}.ply"), pcd, write_ascii=True)
            combined_points.append(obj_points)
            combined_colors.append(obj_colors)
        
        if combined_points and combined_colors:
            combined_points = np.concatenate(combined_points, axis=0)
            combined_colors = np.concatenate(combined_colors, axis=0)
            combined_pcd = o3d.geometry.PointCloud()
            combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
            combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
            combined_pcd = process_pointcloud(combined_pcd)
            o3d.io.write_point_cloud(os.path.join(frame_path, "all_objects.ply"), combined_pcd, write_ascii=True)
        



# 添加主入口调用
if __name__ == "__main__":
    main()
