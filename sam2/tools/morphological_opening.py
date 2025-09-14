import cv2
import numpy as np
import os
import re
import matplotlib
import colorsys
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="/data1/mice/0516", help="path contains videos")
    parser.add_argument("--video_list",
                        default=['play'],
                        help="list of video folders. eg. v1, v2, v3...", )
    parser.add_argument("--device_list", default=["0","1","2","3"], help="list of cameras", )
    parser.add_argument("--object_num", type=int, default=2, help="number of object in the frame", )

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

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

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
def per_obj_mask_opening(per_obj_mask, kernel_size,recover_threshold, filter_threshold):
    # 定义结构元素
    kernel = np.ones(kernel_size, np.uint8)

    #### 进行开运算 ####
    processed_mask = cv2.morphologyEx(per_obj_mask, cv2.MORPH_OPEN, kernel)

    #### 恢复mask细节 ####
    height, width = processed_mask.shape
    detail_mask = cv2.absdiff(per_obj_mask, processed_mask)  # 提取细节差异区域[2](@ref)
    # 连通域分析（含统计信息）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(detail_mask, connectivity=8)
    # 设置面积阈值（单位：像素）
    area_threshold = recover_threshold  # 根据实际效果调整
    for j in range(1, num_labels):  # 跳过背景标签0
        current_area = stats[j, cv2.CC_STAT_AREA]
        if current_area < area_threshold:
            # 获取该区域所有像素坐标 (y, x)
            coords = np.column_stack(np.where(labels == j))
            valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < width) & \
                            (coords[:, 1] >= 0) & (coords[:, 1] < height)
            valid_coords = coords[valid_mask]
            # 步骤2: 提取行列索引（假设coords格式为[x, y]）
            x = valid_coords[:, 0].astype(int)  # 列坐标（宽度方向）
            y = valid_coords[:, 1].astype(int)  # 行坐标（高度方向）
            # 步骤3: 批量设置值
            processed_mask[x, y] = 1

    #### Filter the area below threshold ####
    # 连通域分析（含统计信息）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_mask, connectivity=8)
    # 设置面积阈值（单位：像素）
    for j in range(1, num_labels):  # 跳过背景标签0
        current_area = stats[j, cv2.CC_STAT_AREA]
        if current_area < filter_threshold:
            # 获取该区域所有像素坐标 (y, x)
            coords = np.column_stack(np.where(labels == j))
            valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < width) & \
                            (coords[:, 1] >= 0) & (coords[:, 1] < height)
            valid_coords = coords[valid_mask]
            # 步骤2: 提取行列索引（假设coords格式为[x, y]）
            x = valid_coords[:, 0].astype(int)  # 列坐标（宽度方向）
            y = valid_coords[:, 1].astype(int)  # 行坐标（高度方向）
            # 步骤3: 批量设置值
            processed_mask[x, y] = 0

    return processed_mask
        
def main():
    args = parse_args()
    for video in args.video_list:
        for device_number in args.device_list:
            mask_path = f'{args.video_path}/{video}/device{device_number}/mask'
            for file_name in sorted(os.listdir(f'{mask_path}'), key=natural_sort_key):
                combined_mask, _ = load_ann_png(os.path.join(mask_path, f'{file_name}'))
                height, width = combined_mask.shape[:2]
                all_masks = []
                all_ids = []
                for i in range(1, args.object_num + 1):
                    try:
                        object_mask = get_per_obj_mask(combined_mask)[i].astype(np.uint8)

                        processed_mask = per_obj_mask_opening(object_mask, kernel_size=(13, 13), recover_threshold=40)

                        all_masks.append(processed_mask)
                        all_ids.append(i)
                        # # visualization
                        # plt.subplot(121), plt.imshow(object_mask, cmap='gray')
                        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
                        # plt.subplot(122), plt.imshow(processed_mask, cmap='gray')
                        # plt.title('differences'), plt.xticks([]), plt.yticks([])
                        # plt.show()
                        # # print(object_mask)

                    except KeyError:
                        # 捕获索引超出范围的异常，表示没有这个对象的掩膜
                        print(f"Warning: video {video} device {device_number} frame {file_name} Object {i} does not have a mask. Skipping.")
                        zero_array = np.zeros((height, width),dtype=np.uint8)
                        all_masks.append(zero_array)
                        all_ids.append(i)

                combined_masks = create_multi_mask(all_masks, all_ids)

                saving_path = f'{args.video_path}/{video}/device{device_number}/processed_mask/'
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)
                combined_masks.save(os.path.join(saving_path,f"{file_name}")) # for testing


if __name__ == "__main__":
    main()