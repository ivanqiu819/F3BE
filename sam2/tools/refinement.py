import numpy as np
import os
import sys
import torch
import json
import random
import argparse
import cv2
import re
import shutil
from PIL import Image
import colorsys
from matplotlib import pyplot as plt
from skimage.morphology import erosion, square
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from sam2.build_sam import build_sam2_video_predictor

class Refinement:
    """
    A class to handle the refinement of masks using a SAM model.
    It includes methods for refined masks using reference devices.
    """

    def __init__(
                self,
                base_path: str,
                source_device: str,
                sam2_checkpoint: str,
                sam2_cfg: str,
                config_file: str,
                GPU_device: str,
                visualize: bool,
                depth_threshold: float,
                color_threshold: float
                ):
        """
        Initialize the Refinement class with the SAM model path and device.

        Parameters:
        sam_model_path (str): Path to the SAM model.
        device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.base_path = base_path
        self.source_device = source_device
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_cfg = sam2_cfg
        self.config_file = config_file
        self.device_parameters = json.load(open(config_file, 'r', encoding='utf-8-sig'))
        # self.sam_predictor = build_sam2_video_predictor(sam_model_path, device=GPU_device)
        self.device = GPU_device
        self.visualize = visualize
        self.depth_threshold = depth_threshold
        self.color_threshold = color_threshold


    @staticmethod
    def show_mask(mask, ax, obj_id=None, random_color=False):
        """
        Display the mask with a specified color on the given axis.

        Parameters:
        mask (numpy.ndarray): The binary mask to be displayed.
        ax (matplotlib.axes.Axes): The axis on which to display the mask.
        obj_id (int, optional): The object ID for color selection in a predefined color map.
        random_color (bool, optional): Whether to display the mask with a random color.

        Returns:
        numpy.ndarray: The mask image with applied color.
        """
        # Choose a random color or use a predefined color map
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)  # Random color with transparency
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])  # Default color from colormap

        h, w = mask.shape[-2:]  # Height and width of the mask
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)  # Apply color to the mask
        print(mask_image.shape)
        ax.imshow(mask_image)  # Show the mask image on the provided axis
        return mask_image

    @staticmethod
    def create_multi_mask(masks_list, object_ids):
        """
        生成支持多对象的P模式掩码图像

        参数：
        masks_list (list of np.ndarray): 多个布尔型掩码组成的列表
        object_ids (list of int): 对应掩码的对象ID列表

        返回：
        Image.Image: P模式的掩码图像
        """
        # 参数校验
        if len(masks_list) != len(object_ids):
            raise ValueError("掩码数量与对象ID数量必须一致")
        if max(object_ids) > 255:
            raise ValueError("对象ID最大不能超过255")

        # 获取最大图像尺寸
        max_h = max(m.shape[0] for m in masks_list)
        max_w = max(m.shape[1] for m in masks_list)

        # 初始化全黑背景
        final_mask = np.zeros((max_h, max_w), dtype=np.uint8)

        # 映射对象ID到调色板索引
        unique_ids = sorted(set(object_ids))
        id_to_index = {oid: idx+1 for idx, oid in enumerate(unique_ids)}

        # 合并掩码
        for mask, oid in zip(masks_list, object_ids):
            mask = mask.squeeze()  # 移除所有单维度（如Batch或Channel维度）
            if oid in id_to_index:
                final_mask[mask.astype(bool)] = id_to_index[oid]

        # 生成动态调色板
        palette = [0, 0, 0]  # 背景黑色
        for idx in range(len(unique_ids)):
            hue = idx * (240 / len(unique_ids))  # 0-240°色相范围
            r, g, b = colorsys.hsv_to_rgb(hue/360, 0.9, 0.9)
            palette += [int(r*255), int(g*255), int(b*255)]
        palette += [0] * (768 - len(palette))  # 填充剩余调色板

        # 创建P模式图像
        mask_image = Image.fromarray(final_mask, mode='P')
        mask_image.putpalette(palette[:768])

        return mask_image

    @staticmethod
    def show_points(coords, labels, ax, marker_size=200):
        """
        Display points on a 2D plot, with positive points in green and negative points in red.

        Parameters:
        coords (numpy.ndarray): Coordinates of the points to be displayed (N x 2).
        labels (numpy.ndarray): Labels (0 or 1) to differentiate between positive and negative points.
        ax (matplotlib.axes.Axes): The axis to plot the points.
        marker_size (int, optional): The size of the points to be displayed.
        """
        # Select positive and negative points based on the labels
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]

        # Plot positive points in green and negative points in red
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                linewidth=1.25)

    @staticmethod
    def camera_to_pixel_with_depth(points, cam_intrinsic):
        """
        Project 3D points from the camera coordinate system to 2D pixel coordinates, along with their depth.

        Parameters:
        points (numpy.ndarray): 3D points in the camera coordinate system (N x 3).
        cam_intrinsic (numpy.ndarray): The camera intrinsic matrix (3 x 3).

        Returns:
        numpy.ndarray: Pixel coordinates (u, v) along with their depth (z), shape (N x 3).
        """
        # Convert to homogeneous coordinates (N x 4)
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)

        # Project points to the image plane using the intrinsic matrix
        projected = np.dot(cam_intrinsic, points_homogeneous[:, :3].T).T  # (N, 3)

        # Convert to non-homogeneous coordinates (u, v)
        u = projected[:, 0] / projected[:, 2]
        v = projected[:, 1] / projected[:, 2]

        # Depth value is the z-coordinate in camera space
        z = points[:, 2]  # Depth (z)

        # Stack (u, v, z) as the final result
        pixel_coords_with_depth = np.stack((u, v, z), axis=-1)

        return pixel_coords_with_depth

    @staticmethod
    def mask_depth_map(depth_map, selected_pixels):
        """
        Generate a masked depth map with values from selected pixels.

        Parameters:
        depth_map (numpy.ndarray): The original depth map (H x W).
        selected_pixels (list of tuples): List of (u, v) pixel coordinates.

        Returns:
        numpy.ndarray: A new depth map where only selected pixels retain their values.
        """
        # Create an empty mask with the same shape as the depth map
        masked_depth_map = np.zeros_like(depth_map)

        # Set the depth values at the selected pixels
        for u, v in selected_pixels:
            masked_depth_map[u, v] = depth_map[u, v]

        return masked_depth_map

    @staticmethod
    def depth2pts_np(depth_map, cam_intrinsic, cam_extrinsic=np.eye(4)):
        """
        Convert a depth map to 3D points in world coordinates.

        Parameters:
        depth_map (numpy.ndarray): The depth map (H x W).
        cam_intrinsic (numpy.ndarray): The camera intrinsic matrix (3 x 3).
        cam_extrinsic (numpy.ndarray, optional): The camera extrinsic matrix (4 x 4), defaults to identity matrix.

        Returns:
        numpy.ndarray: 3D points in world coordinates (N x 3).
        """
        # Get pixel grid coordinates
        feature_grid = Refinement.get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

        # Project pixel coordinates to camera coordinates
        uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
        cam_points = uv * np.reshape(depth_map, (1, -1))

        # Apply the camera extrinsics to transform the points into world coordinates
        R = cam_extrinsic[:3, :3]
        t = cam_extrinsic[:3, 3:4]
        R_inv = np.linalg.inv(R)

        # Transform the points from camera space to world space
        world_points = np.matmul(R_inv, cam_points - t).transpose()

        return world_points

    @staticmethod
    def get_pixel_grids_np(height, width):
        """
        Generate a grid of pixel coordinates (u, v).

        Parameters:
        height (int): Height of the image.
        width (int): Width of the image.

        Returns:
        numpy.ndarray: A grid of pixel coordinates (3, H * W).
        """
        # Create linearly spaced coordinates for x and y
        x_linspace = np.linspace(0.5, width - 0.5, width)
        y_linspace = np.linspace(0.5, height - 0.5, height)
        x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)

        # Reshape the coordinates into 1D arrays and stack with ones for homogeneous coordinates
        x_coordinates = np.reshape(x_coordinates, (1, -1))
        y_coordinates = np.reshape(y_coordinates, (1, -1))
        ones = np.ones_like(x_coordinates).astype(float)

        grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)  # (3, H * W)

        return grid

    @staticmethod
    def filter_prompts(
            current_depth_map,
            prompt_list,
            rgb_image,
            depth_threshold=0.01,
            brightness_threshold=100,
            grid_size=20,
            max_points_per_grid=2
    ):
        """
        Filter points based on depth difference and brightness, then perform grid-based downsampling.

        Parameters:
        current_depth_map (numpy.ndarray): The current depth map (H x W).
        rgb_image (numpy.ndarray): The RGB image (H x W x 3).
        prompt_list (list): A list of prompts with (x, y, depth_comparison_value).
        threshold (float, optional): Depth difference threshold, default is 0.01.
        brightness_threshold (float, optional): Brightness threshold, default is 100.
        grid_size (int, optional): Size of the grid for downsampling, default is 20.
        max_points_per_grid (int, optional): Max points to sample per grid, default is 2.

        Returns:
        list: A list of filtered and downsampled points.
        """
        all_results = []

        for sublist in prompt_list:
            results = []

            for prompt in sublist:
                x, y, reference_depth = prompt

                # Ensure the pixel is within bounds of the depth map
                if 0 <= x < current_depth_map.shape[1] and 0 <= y < current_depth_map.shape[0]:
                    depth_from_map = current_depth_map[int(y), int(x)]
                    depth_difference = depth_from_map - reference_depth

                    # Compute brightness of the pixel
                    r, g, b = rgb_image[int(y), int(x)]
                    brightness = 0.299 * r + 0.587 * g + 0.114 * b

                    # Apply the filtering conditions
                    if abs(depth_difference) <= depth_threshold and brightness < brightness_threshold:
                        results.append([x, y])

            # Convert results to NumPy array for further processing
            results_array = np.array(results, dtype=float)

            # Apply grid-based downsampling if there are results
            if len(results_array) > 0:
                grid_dict = {}

                # Group points by grid
                for point in results_array:
                    grid_x = int(point[0] // grid_size)
                    grid_y = int(point[1] // grid_size)
                    grid_key = (grid_x, grid_y)

                    if grid_key not in grid_dict:
                        grid_dict[grid_key] = []
                    grid_dict[grid_key].append(point)

                # Sample points from each grid (limit max points per grid)
                sampled_points = []
                for grid_key, points in grid_dict.items():
                    if len(points) > max_points_per_grid:
                        sampled_points.extend(random.sample(points, max_points_per_grid))
                    else:
                        sampled_points.extend(points)

                results_array = np.array(sampled_points, dtype=float)

            all_results.append(results_array)

        return all_results

    @staticmethod
    def filter_and_round_with_depth(points):
        """
        Filter out points with negative x or y values and round the x, y coordinates.

        Parameters:
        points (numpy.ndarray): List of 3D points with (x, y, z).

        Returns:
        numpy.ndarray: Filtered and rounded points.
        """
        filtered_points = [
            [int(round(row[0])), int(round(row[1])), row[2]]  # Round x, y and keep z
            for row in points if row[0] >= 0 and row[1] >= 0
        ]

        # Convert to NumPy array and return
        filtered_points = np.array(filtered_points)

        return filtered_points

    @staticmethod
    def load_ann_png(path):
        """
        Load a PNG mask image and extract its color palette.

        Parameters:
        path (str): The path to the PNG file.

        Returns:
        numpy.ndarray: The mask image as a NumPy array.
        list: The color palette used in the mask.
        """
        mask = Image.open(path)
        palette = mask.getpalette()
        mask = np.array(mask).astype(np.uint8)
        return mask, palette

    @staticmethod
    def get_points_on_eroded_mask(object_mask):
        """
        Perform morphological erosion on the mask and extract the points.

        Parameters:
        object_mask (numpy.ndarray): The binary mask to be processed.

        Returns:
        numpy.ndarray: The points after applying erosion.
        """
        # Apply morphological erosion with a square kernel of size 15
        eroded_mask = erosion(object_mask, square(15))  # Shrink the mask

        # Extract coordinates of the eroded mask (points where the mask is 1)
        prompts = np.round(np.column_stack(np.where(eroded_mask == 1))).astype(int)

        return prompts

    # Function to generate reference prompts based on device IDs and paths to the reference data
    def get_reference_prompt(self, source_device, reference_device, reference_mask_path, reference_depth_path):
        # Define intrinsic camera matrices for each device
        source_device_intrinsic = self.device_parameters[source_device]['stereo_params']['intrinsic']
        reference_device_intrinsic = self.device_parameters[reference_device]['stereo_params']['intrinsic']

        # Define extrinsic camera matrices to convert between different device coordinate systems
        reference_wrt_source_transform = np.array(self.device_parameters[reference_device][f"{reference_device}_wrt_{source_device}"])

        combined_prompts = []
        # Get reference mask
        current_mask, _ = self.load_ann_png(reference_mask_path)
        reference_depth_map = cv2.imread(reference_depth_path, cv2.IMREAD_UNCHANGED).astype(float) / 1000.0

        for i in range(1,current_mask.max()+1):
            object_mask = (current_mask == i).astype(np.uint8)

            # Get side view prompts (u,v coordinate)
            object_prompts = self.get_points_on_eroded_mask(object_mask)

            # Get depths of side view prompts(z coordinate)
            filtered_map = self.mask_depth_map(reference_depth_map, object_prompts)
            
            # Get xyz coordinates of side view prompts & Convert coordinates system from side view device to current device

            object_prompts = self.depth2pts_np(filtered_map, reference_device_intrinsic, np.linalg.inv(reference_wrt_source_transform))
            object_prompts = self.camera_to_pixel_with_depth(object_prompts, source_device_intrinsic)

            # Filter out the points which have u/v coordinates below zero
            combined_prompts.append(self.filter_and_round_with_depth(object_prompts))

        return combined_prompts
    
    @staticmethod
    def natural_sort_key(s):
        """
        Sort filenames or strings that contain numbers in a natural order.

        Args:
            s (str): The string to be sorted.

        Returns:
            list: A list of numbers and strings, split for natural sorting.
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    
    def refine(self):
        APPLY_POSTPROCESSING = True
        USE_VOS_OPTIMIZED_VIDEO_PREDICTOR = False
        PER_OBJ_PNG_FILE = False
        # if we use per-object PNG files, they could possibly overlap in inputs and outputs
        hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if PER_OBJ_PNG_FILE else "true")]
        predictor = build_sam2_video_predictor(
                                            config_file = self.sam2_cfg, 
                                            ckpt_path= self.sam2_checkpoint, 
                                            device=self.device,
                                            apply_postprocessing=APPLY_POSTPROCESSING,
                                            hydra_overrides_extra=hydra_overrides_extra,
                                            vos_optimized=USE_VOS_OPTIMIZED_VIDEO_PREDICTOR)
        refine_list = f'{self.base_path}/{self.source_device}/refine/'
        refine_names = sorted(os.listdir(refine_list),key=self.natural_sort_key)
        # refine_names = [12180]
        for refine_name in refine_names:
            # source_mask_path = f'{self.base_path}/{self.source_device}/mask/{refine_name}.png'
            source_depth_path = f'{self.base_path}/{self.source_device}/depth/{refine_name}.png'
            current_depth_map = cv2.imread(source_depth_path, cv2.IMREAD_UNCHANGED).astype(float) / 1000.0
            rgb_image = cv2.imread(f"{self.base_path}/{self.source_device}/rect_left_camera/{refine_name}.jpg")
            refine_dir = f"{self.base_path}/{self.source_device}/refine/{refine_name}"
            # Remove other images
            for filename in os.listdir(refine_dir):
                file_path = os.path.join(refine_dir, filename)
                # 检查是否为文件
                if os.path.isfile(file_path):
                    # 获取文件扩展名
                    _, file_ext = os.path.splitext(filename)
                    # 检查文件是否不符合要求（不以 refine_name 开头或者扩展名不在允许列表中）
                    if not filename.startswith(refine_name) or file_ext.lower() not in [ext.lower() for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]]:
                        try:
                            # 尝试删除文件
                            os.remove(file_path)
                            print(f"Removed: {file_path}")
                        except Exception as e:
                            print(f"Error removing {file_path}: {e}")

            # Get reference device_ID according to current device
            # current_device, reference_device_1, reference_device_2 = get_reference_device_num(current_mask_path)
            reference_devices = list(self.device_parameters[self.source_device]["reference_devices"])
            total_reference_prompts = {}
            for idx, reference_device in enumerate(reference_devices):
                # reference_prompts[reference_device] = {}
                reference_mask_path = f'{self.base_path}/{reference_device}/mask/{refine_name}.png'
                reference_depth_path = f'{self.base_path}/{reference_device}/depth/{refine_name}.png'

                reference_prompts = self.get_reference_prompt(self.source_device, reference_device, reference_mask_path,reference_depth_path)
                reference_prompts = self.filter_prompts(current_depth_map=current_depth_map,
                                                         prompt_list=reference_prompts, 
                                                         rgb_image=rgb_image, 
                                                         depth_threshold=self.depth_threshold, 
                                                         brightness_threshold=self.color_threshold, 
                                                         grid_size=20,
                                                         max_points_per_grid=1)
                total_reference_prompts[reference_device] = reference_prompts

                if self.visualize:
                    # # visualization
                    colors = ['red', 'green', 'blue', 'orange', 'purple']
                    plt.figure(figsize=(12, 6))
                    plt.axis('off')
                    plt.imshow(rgb_image)

                    # 遍历 reference_prompts，并为每个子列表分配不同颜色
                    for idx, sublist in enumerate(reference_prompts):
                        color = colors[idx % len(colors)]  # 循环使用颜色列表
                        for point in sublist:
                            x, y = point
                            plt.scatter([x], [y], s=1, c=color)  # 设置当前子列表的颜色
                    plt.show()
                    print('prompt done')

            ### combine prompts from reference devices
            # 1. 确定 mask 数量
            mask_count = len(next(iter(total_reference_prompts.values())))

            # 2. 聚合结果列表
            mask_prompts = []

            for idx in range(mask_count):
                # 收集各 device 在第 idx 个 mask 的点集
                arrays = [total_reference_prompts[dev][idx].reshape(-1, 2) for dev in total_reference_prompts if len(total_reference_prompts[dev]) > idx]
                # 垂直拼接，若全为空也会得到 shape=(0,2) 的数组
                merged = np.vstack(arrays)
                mask_prompts.append(merged)

            # aggregated_prompts 是一个 list，len = mask_count
            # 其中 aggregated_prompts[i] 即为所有 device 在 mask i 上的点集
            for i, pts in enumerate(mask_prompts):
                print(f"Mask {i}: shape = {pts.shape}")
            '''
            prompts = []
            for i in range(max(len(reference_prompts_1),len(reference_prompts_2))):
                # Check if either reference_prompts_1[i] or reference_prompts_2[i] is an empty list or array
                if reference_prompts_1[i].size == 0:
                    object_prompt = reference_prompts_2[i]
                elif reference_prompts_2[i].size == 0:
                    object_prompt = reference_prompts_1[i]
                else:
                    # If both are not empty, concatenate them
                    object_prompt = np.concatenate((reference_prompts_1[i], reference_prompts_2[i]), axis=0)

                prompts.append(object_prompt)
            '''
            

            # use sam2 to refine images

            # scan all the JPEG frame names in this directory
            frame_names = [
                p for p in os.listdir(refine_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            inference_state = predictor.init_state(video_path=refine_dir)
            predictor.reset_state(inference_state)
            dict = {}
            processed_masks = None
            for i in range(len(mask_prompts)):
                ann_frame_idx = 0  # the frame index we interact with
                ann_obj_id = i+1  # give a unique id to each object we interact with (it can be any integers)
                positive_points = mask_prompts[i]
                positive_labels = np.ones((positive_points.shape[0]), dtype=int)
                neg_points_list = [mask_prompts[j] for j in range(len(mask_prompts)) if j != i and mask_prompts[j].size > 0]
                if len(neg_points_list) > 0:
                    negative_points = np.concatenate(neg_points_list, axis=0)
                else:
                    negative_points = np.empty((0, 2))
                negative_labels = np.zeros((negative_points.shape[0]), dtype=int)
                if positive_points.size == 0:
                    points = negative_points
                elif negative_points.size == 0:
                    points = positive_points
                else:
                    points = np.concatenate((positive_points, negative_points), axis=0)

                if positive_labels.size == 0:
                    labels= negative_labels
                elif negative_labels.size == 0:
                    labels= positive_labels
                else:
                    labels = np.concatenate((positive_labels, negative_labels), axis=0)
                dict[ann_obj_id] = points, labels
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )

            # Post-processing step to refine the generated masks
            for i, out_obj_id in enumerate(out_obj_ids):
                # Convert logits to binary mask where values greater than 0 are set to 1, else 0
                processing_mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8)

                # Define a structuring element (kernel) for morphological operations - using an elliptical kernel of size 5x5
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)) # 改写成参数传递

                # Apply morphological closing operation: first dilate, then erode to remove small holes in the mask
                processed_mask = cv2.morphologyEx(processing_mask[0, :, :], cv2.MORPH_CLOSE, kernel).astype(bool)

                # If it's the first processed mask, initialize the processed_masks array
                if processed_masks is None:
                    processed_masks = processed_mask[np.newaxis, ...]
                else:
                    # Otherwise, concatenate the new processed mask with the previous ones
                    processed_masks = np.concatenate((processed_masks, processed_mask[np.newaxis, ...]), axis=0)
            if self.visualize:
                # visualization
                plt.figure(figsize=(9, 6))
                plt.title(f"frame {ann_frame_idx}")
                plt.imshow(Image.open(os.path.join(refine_dir, frame_names[ann_frame_idx])))
                self.show_points(points, labels, plt.gca())
                for i, out_obj_id in enumerate(out_obj_ids):
                    self.show_points(*dict[out_obj_id], plt.gca())
                    self.show_mask(processed_masks[i], plt.gca(), obj_id=out_obj_id)
                plt.show()

            ### Create Annotation

            # Initialize lists to store all the masks and their corresponding object IDs
            all_masks = []
            all_ids = []

            # Create a dictionary of image segments with object IDs as keys and the processed masks as values
            image_segments = {
                out_obj_id: processed_masks[i]
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            # Loop through the image segments and append each mask and its associated object ID to the lists
            for out_obj_id, out_mask in image_segments.items():
                all_masks.append(out_mask)
                all_ids.append(out_obj_id)

            # Create a combined mask image from all individual masks and their object IDs
            combined_masks = self.create_multi_mask(all_masks, all_ids)

            # clean the refine directory by removing all files
            for filename in os.listdir(refine_dir):
                file_path = os.path.join(refine_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            # Define the output path for saving the combined mask image
            output_path = os.path.join(refine_dir, f"{refine_name}.png")

            # Save the combined mask image to the output path
            combined_masks.save(output_path)

            # Reset the predictor's state to free up resources and avoid potential memory issues
            predictor.reset_state(inference_state)

            # Delete the inference state object to release memory
            del inference_state

            # Clear the GPU memory cache to free up unused memory
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Process mask files.')
    parser.add_argument('--base_path', type=str, default='/mnt/disk0/mice', help='base path containing the source device and reference devices')
    parser.add_argument('--source_device',type=str, default="device0", help='the device to be refined')
    parser.add_argument('--sam2_checkpoint', type=str, default='../checkpoints/sam2.1_hiera_large.pt', help='Path to the SAM2 checkpoint')
    parser.add_argument('--model_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml', help='Path to the SAM2 model configuration file')
    parser.add_argument('--config_file', type=str, default='', help='config file for devices(cameras)')
    parser.add_argument("--GPU_device", type=str, default="cuda:1", help="Torch device")
    parser.add_argument('--visualize', default= False, help='Show visualization of refined masks')
    args = parser.parse_args()

    # Initialization
    if torch.cuda.is_available():
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    refiner = Refinement(base_path=args.base_path, 
                         source_device=args.source_device, 
                         sam2_checkpoint=args.sam2_checkpoint,
                         model_cfg=args.model_cfg,
                         config_file=args.config_file,
                         GPU_device=args.GPU_device, 
                         visualize=args.visualize)

if __name__ == '__main__':
    main()