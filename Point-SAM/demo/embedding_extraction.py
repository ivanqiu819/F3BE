import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from point_sam import build_point_sam
import torch
import torch.nn as nn
import argparse



def natural_sort_key(s):
    """
    Sorts strings in a human-readable way, sorting numbers in the string numerically.
    """
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_ply(filename):
    with open(filename, "r") as rf:
        while True:
            try:
                line = rf.readline()
            except:
                raise NotImplementedError
            if "end_header" in line:
                break
            if "element vertex" in line:
                arr = line.split()
                num_of_points = int(arr[2])

        # print("%d points in ply file" %num_of_points)
        points = np.zeros([num_of_points, 6])
        for i in range(points.shape[0]):
            point = rf.readline().split()
            assert len(point) == 6
            points[i][0] = float(point[0])
            points[i][1] = float(point[1])
            points[i][2] = float(point[2])
            points[i][3] = float(point[3])
            points[i][4] = float(point[4])
            points[i][5] = float(point[5])
    rf.close()
    del rf
    return points

def load_pointcloud(pointcloud_path):
    """
    Loads point cloud data from a .ply file and normalizes it.
    """
    points = load_ply(pointcloud_path)
    xyz = points[:, :3]
    rgb = points[:, 3:6] / 255.0

    pc_xyz, pc_rgb = torch.from_numpy(xyz).cuda().float(), torch.from_numpy(rgb).cuda().float()
    pc_xyz, pc_rgb = pc_xyz.unsqueeze(0), pc_rgb.unsqueeze(0)

    return pc_xyz, pc_rgb


def rotate_point_cloud_z_axis(point_cloud, angle_degrees=90):
    """
    Rotates a point cloud around the Z-axis by a specified angle.
    """
    angle_radians = np.radians(angle_degrees)
    rotation_matrix_z = torch.tensor([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians),  np.cos(angle_radians), 0],
        [0,                     0,                    1]
    ], dtype=point_cloud.dtype, device=point_cloud.device)

    rotation_matrix_z = rotation_matrix_z.unsqueeze(0)
    rotated_point_cloud = torch.matmul(point_cloud, rotation_matrix_z)

    return rotated_point_cloud


def main(args):
    sam = build_point_sam(args.model_path).cuda()

    rotate_num = args.rotate_num
    pointcloud_dir = os.path.join(args.base_path, 'pointclouds')
    embedding_path = os.path.join(args.base_path, 'embeddings')
    os.makedirs(embedding_path, exist_ok=True)
    frames = sorted(os.listdir(pointcloud_dir), key=natural_sort_key)
    print(f"Processing {len(frames)} frames...")
    for frame in frames:
        print(f"Processing frame: {frame}")
        rotated_pc_list = []
        frame_path = os.path.join(pointcloud_dir, frame)
        # Load point cloud and images
        pointcloud_path = os.path.join(frame_path, "all_objects.ply")
        if not os.path.exists(pointcloud_path):
            print(f"Point cloud file {pointcloud_path} does not exist. Skipping frame {frame}.")
            continue

        pc_xyz, pc_rgb = load_pointcloud(pointcloud_path)

        if args.centralize:
            # Get the centroid of the point cloud
            centroid = (torch.max(pc_xyz, dim=1, keepdim=True).values +
                        torch.min(pc_xyz, dim=1, keepdim=True).values) / 2
            pc_xyz = pc_xyz - centroid

        if args.normalize:
            # Normalize by the maximum distance
            max_distance = torch.max(torch.norm(pc_xyz, dim=2))
            pc_xyz = pc_xyz / max_distance

        # Apply rotation
        for i in range(args.rotate_num):
            rotated_pc_xyz = rotate_point_cloud_z_axis(pc_xyz, angle_degrees=360 / args.rotate_num * i)
            rotated_pc_list.append(rotated_pc_xyz)

        # Concatenate all rotated point clouds
        rotated_pc_sum = torch.cat(rotated_pc_list, dim=0)

        with torch.no_grad():
            # Expand color data to match rotated point clouds
            if pc_rgb.dim() == 2:
                pc_rgb = pc_rgb.unsqueeze(0)
            pc_rgb_expanded = pc_rgb.repeat(rotate_num, 1, 1)

            # Process the point cloud with SAM
            sam.set_pointcloud(rotated_pc_sum, pc_rgb_expanded)
            max_pool_layer_1 = nn.AdaptiveMaxPool2d((1, 256))
            sam.pc_embeddings = max_pool_layer_1(sam.pc_embeddings)

            # Apply average pooling across the batch dimension
            if args.pooling_type == "mean":
                sam.pc_embeddings = torch.mean(sam.pc_embeddings, dim=0, keepdim=True)
            elif args.pooling_type == "max":
                sam.pc_embeddings = torch.max(sam.pc_embeddings, dim=0, keepdim=True).values #max pooling
     
        out_dir = os.path.join(embedding_path, frame)
        os.makedirs(out_dir, exist_ok=True)
        embedding_name = f"norm={args.normalize}_center={args.centralize}_{args.rotate_num}feature_{args.pooling_type}.pth"
        torch.save(sam.pc_embeddings, f"{out_dir}/{embedding_name}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point Cloud Processing Pipeline")
    # Basic arguments
    parser.add_argument('--base_path', type=str,
                         default="/home/user/PycharmProjects/animal_pose_-gui/example_operator1/experiment/test",
                         help="Base path for point cloud and images.")
    parser.add_argument('--model_path', type=str, 
                        default="/home/user/Point-SAM/model.safetensors", 
                        help="Path to the Point-SAM model.")
    parser.add_argument('--rotate_num', type=int, default=16, help="Number of rotations to apply on each point cloud.")
    parser.add_argument('--centralize', type=bool, default=True, help="Whether to centralize the point cloud.")
    parser.add_argument('--normalize', type=bool, default=True, help="Whether to normalize the point cloud.")
    parser.add_argument("--pooling_type", type=str, default="max", choices=["max", "mean"], help="Pooling type.")
    args = parser.parse_args()

    main(args)