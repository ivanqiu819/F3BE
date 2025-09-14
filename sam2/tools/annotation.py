# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Video segmentation with SAM 2

This script demonstrates how to use SAM 2 for interactive segmentation in videos. It covers:

- Adding clicks (or box) on a frame to get and refine masklets (spatio-temporal masks)
- Propagating clicks (or box) to get masklets throughout the video
- Segmenting and tracking multiple objects at the same time

We use the terms segment or mask to refer to the model prediction for an object on a single frame, 
and masklet to refer to the spatio-temporal masks across the entire video.
"""
import sys
import os
import shutil
import numpy as np
import torch
import cv2
import colorsys
import argparse
from utils import add_depth_gradient_to_rgb
from PIL import Image
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from sam2.build_sam import build_sam2_video_predictor
import pathlib

# Enable MPS fallback for Apple devices and set CUDA device
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup_device():
    """Set up computation device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    return device

def create_multi_mask(masks_list, object_ids):
    """
    Create a multi-object mask image in 'P' mode.

    Args:
        masks_list (list of np.ndarray): List of boolean masks.
        object_ids (list): List of object IDs (int or str).

    Returns:
        Image.Image: Mask image in 'P' mode.
    """
    if len(masks_list) != len(object_ids):
        raise ValueError("Number of masks and object IDs must match.")
    # Only check max for integer IDs
    int_object_ids = [oid for oid in object_ids if isinstance(oid, int)]
    if int_object_ids and max(int_object_ids) > 255:
        raise ValueError("Object ID must not exceed 255.")

    max_h = max(m.shape[1] for m in masks_list)
    max_w = max(m.shape[2] for m in masks_list)
    final_mask = np.zeros((max_h, max_w), dtype=np.uint8)

    # Only use integer IDs for palette and mask assignment
    unique_ids = sorted(set(int_object_ids))
    id_to_index = {oid: idx+1 for idx, oid in enumerate(unique_ids)}

    for mask, oid in zip(masks_list, object_ids):
        mask = mask.squeeze()
        if isinstance(oid, int) and oid in id_to_index:
            final_mask[mask.astype(bool)] = id_to_index[oid]

    palette = [0, 0, 0]
    for idx in range(len(unique_ids)):
        hue = idx * (240 / len(unique_ids))
        r, g, b = colorsys.hsv_to_rgb(hue/360, 0.9, 0.9)
        palette += [int(r*255), int(g*255), int(b*255)]
    palette += [0] * (768 - len(palette))

    mask_image = Image.fromarray(final_mask, mode='P')
    mask_image.putpalette(palette[:768])

    return mask_image

def get_frame_names(video_dir):
    """Get sorted frame file names from a directory."""
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names

def process_clicks_to_sam_format(clicks):
    """
    Convert click list to SAM2 format.

    Args:
        clicks (list): List of (x, y, mouse_id).

    Returns:
        dict: mouse_id -> (points, labels)
    """
    mouse_clicks = {}
    global_negative_points = []

    for x, y, mouse_id in clicks:
        if mouse_id == 0:
            global_negative_points.append((x, y))
        else:
            if mouse_id not in mouse_clicks:
                mouse_clicks[mouse_id] = []
            mouse_clicks[mouse_id].append((x, y))

    result = {}
    for target_mouse_id, target_points_list in mouse_clicks.items():
        if len(target_points_list) > 0:
            all_points = target_points_list.copy()
            all_labels = [1] * len(target_points_list)
            for other_mouse_id, other_points_list in mouse_clicks.items():
                if other_mouse_id != target_mouse_id:
                    all_points.extend(other_points_list)
                    all_labels.extend([0] * len(other_points_list))
            if global_negative_points:
                all_points.extend(global_negative_points)
                all_labels.extend([0] * len(global_negative_points))
            points = np.array(all_points, dtype=np.float32)
            labels = np.array(all_labels, dtype=np.int32)
            result[target_mouse_id] = (points, labels)
    result['global_negative'] = (np.array(global_negative_points, dtype=np.float32), np.zeros(len(global_negative_points), dtype=np.int32))
    return result

def interactive_click_selection(video_dir, frame_idx:str):
    """
    Interactive tool for selecting points on an image.

    Args:
        video_dir (str): Directory containing frames.
        frame_idx (int): Frame index to annotate.

    Returns:
        list: List of (x, y, mouse_id) clicks.
    """
    img = cv2.imread(f'{video_dir}/{frame_idx}.jpg')
    img_original = img.copy()
    current_id = 1
    clicks = []
    mouse_colors = {
        0: (128, 128, 128), # Gray for background
        1: (0, 0, 255),     # Green
        2: (0, 255, 0),     # Blue
        3: (255, 0, 0),     # Red
        4: (255, 255, 0)    # Cyan
    }

    window_name = "Annotation Tool: Click to add points, Press '1-4' to switch ID, '0' for background, 'r' to reset, 'q' to quit"
    cv2.namedWindow(window_name)

    def show_img_with_text(text:str):
        """Display image with text overlay."""
        img_with_text = img.copy()
        cv2.putText(img_with_text, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.imshow(window_name, img_with_text)
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y, current_id))
            color = mouse_colors[current_id]
            cv2.circle(img, (x, y), 3, color, 1)
            show_img_with_text(f"Click: [{x},{y}], ID: {current_id}")

    cv2.setMouseCallback(window_name, on_EVENT_LBUTTONDOWN)
    show_img_with_text("Start clicking to add points")

    while True:
        # cv2.imshow(window_name, img)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('0'):
            current_id = 0
            show_img_with_text(f"Switched to background ID: {current_id}")
        elif key == ord('1'):
            current_id = 1
            show_img_with_text(f"Switched to ID: {current_id}")
        elif key == ord('2'):
            current_id = 2
            show_img_with_text(f"Switched to ID: {current_id}")
        elif key == ord('3'):
            current_id = 3
            show_img_with_text(f"Switched to ID: {current_id}")
        elif key == ord('4'):
            current_id = 4
            show_img_with_text(f"Switched to ID: {current_id}")
        elif key == ord('r'):
            clicks.clear()
            img = img_original.copy()
            show_img_with_text("Reset all points")
    cv2.destroyAllWindows()
    return clicks

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SAM2 Video Annotation Tool")
    parser.add_argument("--device_dir", type=str, required=True, help="Path to the video directory")
    parser.add_argument("--sam2_checkpoint", type=str,
                        required=True, 
                        help="Path to the SAM2 checkpoint")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Path to the SAM2 model configuration")
    args = parser.parse_args()
    # check if it is absolute path
    if not os.path.isabs(args.device_dir):
        raise ValueError("device_dir must be an absolute path")

    # Set up device
    device = setup_device()

    # Build predictor
    predictor = build_sam2_video_predictor(args.model_cfg, args.sam2_checkpoint, device=device)


    # find the first frame index
    rgb_path = f"{args.device_dir}/left_camera_rectified"
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB path does not exist: {rgb_path}")
    # list all the images in the directory and sort them 
    rgb_path = pathlib.Path(rgb_path)
    image_names = list(rgb_path.glob("*.png")) + list(rgb_path.glob("*.jpg")) + list(rgb_path.glob("*.jpeg"))
    image_names = sorted(image_names)
    annotation_index = str(image_names[0]).split("/")[-1].split(".")[0]

    # Video path config
    video_temp_dir = f"{args.device_dir}/annotation_temp_folder"

    # Prepare temp folder and gradient image
    if os.path.exists(video_temp_dir):
        shutil.rmtree(video_temp_dir)
    os.makedirs(video_temp_dir)
    add_depth_gradient_to_rgb(
        idx=annotation_index,
        depth_path=f"{args.device_dir}/depth",
        rgb_path=f"{args.device_dir}/left_camera_rectified",
        save_path=video_temp_dir,
        threshold=0.7,
        kernel_size=3,
        scale=4,
    )

    # Interactive click selection
    clicks = interactive_click_selection(video_temp_dir, annotation_index)
    print(f"Collected {len(clicks)} clicks")
    for i, (x, y, mouse_id) in enumerate(clicks):
        print(f"Click {i+1}: ({x}, {y}), Mouse ID {mouse_id}")

    # Convert clicks to SAM2 format
    mouse_configs = process_clicks_to_sam_format(clicks)
    print(f"Processed object configs: {list(mouse_configs.keys())}")

    # Print details for each mouse
    for mouse_id, (points, labels) in mouse_configs.items():
        positive_count = np.sum(labels == 1)
        negative_count = np.sum(labels == 0)
        print(f"Mouse {mouse_id}: {positive_count} positive, {negative_count} negative, {len(points)} total")

    # Get frame names
    frame_names = get_frame_names(video_temp_dir)
    print(f"Found {len(frame_names)} frames")

    # Initialize inference state
    print("Initializing inference state...")
    inference_state = predictor.init_state(video_path=video_temp_dir)
    predictor.reset_state(inference_state)

    # Store all prompts
    prompts = {}
    objects_config = [(mouse_id, points, labels) for mouse_id, (points, labels) in mouse_configs.items()]
    ann_frame_idx = 0

    for obj_id, points, labels in objects_config:
        # Skip global_negative or empty points
        if isinstance(obj_id, str) or points is None or len(points) == 0:
            continue
        prompts[obj_id] = (points, labels)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

    # Propagate through the video
    print("Propagating through video...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Save annotation masks
    print("Saving annotation masks...")
    vis_frame_stride = 1
    saving_path = f"{args.device_dir}/annotation"
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        all_masks = []
        all_ids = []
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            all_masks.append(out_mask)
            all_ids.append(out_obj_id)
        combined_masks = create_multi_mask(all_masks, all_ids)
        filename, extension = os.path.splitext(frame_names[out_frame_idx])
        output_path = os.path.join(saving_path, f"{filename}.png")
        combined_masks.save(output_path)

    # Cleanup
    predictor.reset_state(inference_state)
    del inference_state
    torch.cuda.empty_cache()
    shutil.rmtree(video_temp_dir)

    print("Annotation completed!")

if __name__ == "__main__":
    main()