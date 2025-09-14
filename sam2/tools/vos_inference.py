# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import cv2
import os
from collections import defaultdict
import math
import shutil
import numpy as np
import torch
from morphological_opening import per_obj_mask_opening
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from mask_detection import MaskDetection
import matplotlib.pyplot as plt
from typing import Optional
import tqdm
from utils import img_slider_viewer

from typing import Dict


# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette

def load_ann_jpg(path):
    image = Image.open(path)
    image = image.convert('RGB')
    color_info = np.array(image).astype(np.uint8)
    mask = image.convert('L')
    mask = np.array(mask).astype(np.uint8)
    return mask, color_info

def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)

def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask

def put_per_obj_mask(per_obj_mask: Dict[int, np.ndarray], height: int, width: int) -> np.ndarray:
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id].astype(bool)
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask

def load_masks_from_dir(
    video_dataset_path: str,
    input_mask_dir_under_video: str,
    frame_name: str,
    per_obj_png_file: bool,
    allow_missing: bool = False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        mask_dir = os.path.join(video_dataset_path, input_mask_dir_under_video)

        if os.path.exists(os.path.join(mask_dir, f"{frame_name}.png")):
            input_mask_path = os.path.join(mask_dir, f"{frame_name}.png")
            input_mask, input_palette = load_ann_png(input_mask_path)

        elif allow_missing:
            return {}, None
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        raise NotImplementedError("Per-object PNG files are not supported yet")

        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette

def save_masks_to_dir(
    video_dataset_path: str,
    output_mask_dir_under_video: str,
    frame_name: str,
    per_obj_output_mask,
    height: int,
    width: int,
    per_obj_png_file: bool,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    output_dir = os.path.join(video_dataset_path, output_mask_dir_under_video)
    os.makedirs(output_dir, exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(output_dir, f"{frame_name}.png")
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        raise NotImplementedError("Per-object PNG files are not supported yet")

        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)

def detection_dummy_function() -> bool:
    """Check whether the prediction is correct (i.e. no missing or mixed objects)"""
    return True

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_separate_inference_per_object(
    predictor,
    video_dataset_path: str,
    input_mask_dir_under_video: str,
    output_mask_dir_under_video: str,
    batch_id: str,
    opening_kernel_size,
    opening_recover_threshold,
    filter_threshold,
    missing_threshold,
    mixing_threshold,
    object_number,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
) -> int:
    """
    Run VOS inference on a single video with the given predictor.

    Unlike `vos_inference`, this function run inference separately for each object
    in a video, which could be applied to datasets like LVOS or YouTube-VOS that
    don't have all objects to track appearing in the first frame (i.e. some objects
    might appear only later in the video).

    Returns the last successful frame number
    """
    # load the video frames and initialize the inference state on this video
    video_dir = os.path.join(video_dataset_path, "vos_rgb", batch_id)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    # collect all the object ids and their input masks
    inputs_per_object = defaultdict(dict)
    for idx, name in enumerate(frame_names):
        if per_obj_png_file or \
           os.path.exists(os.path.join(video_dataset_path, input_mask_dir_under_video, f"{name}.png")):
            per_obj_input_mask, input_palette = load_masks_from_dir(
                video_dataset_path=video_dataset_path,
                input_mask_dir_under_video=input_mask_dir_under_video,
                frame_name=frame_names[idx],
                per_obj_png_file=per_obj_png_file,
                allow_missing=True,
            )
            for object_id, object_mask in per_obj_input_mask.items():
                # skip empty masks
                if not np.any(object_mask):
                    continue
                # if `use_all_masks=False`, we only use the first mask for each object
                if len(inputs_per_object[object_id]) > 0 and not use_all_masks:
                    continue
                print(f"adding mask from frame {idx} as input for {object_id=}")
                inputs_per_object[object_id][idx] = object_mask

    # run inference separately for each object in the video
    object_ids = sorted(inputs_per_object)
    output_scores_per_object = defaultdict(dict)
    for object_id in object_ids:
        # add those input masks to SAM 2 inference state before propagation
        input_frame_inds = sorted(inputs_per_object[object_id])
        predictor.reset_state(inference_state)
        for input_frame_idx in input_frame_inds:
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=inputs_per_object[object_id][input_frame_idx],
            )


        # run propagation throughout the video and collect the results in a dict
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=min(input_frame_inds),
            reverse=False,
        ):
            obj_scores = out_mask_logits.cpu().numpy()
            output_scores_per_object[object_id][out_frame_idx] = obj_scores

    # post-processing: consolidate the per-object scores into per-frame masks
    output_dir = os.path.join(video_dataset_path, output_mask_dir_under_video)
    os.makedirs(output_dir, exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for frame_idx in range(len(frame_names)):
        scores = torch.full(
            size=(len(object_ids), 1, height, width),
            fill_value=-1024.0,
            dtype=torch.float32,
        )
        for i, object_id in enumerate(object_ids):
            if frame_idx in output_scores_per_object[object_id]:
                scores[i] = torch.from_numpy(
                    output_scores_per_object[object_id][frame_idx]
                )

        if not per_obj_png_file:
            scores = predictor._apply_non_overlapping_constraints(scores)
        per_obj_output_mask = {
            object_id: (scores[i] > score_thresh).cpu().numpy()
            for i, object_id in enumerate(object_ids)
        }
        video_segments[frame_idx] = per_obj_output_mask
    predictor.reset_state(inference_state)
    del inference_state, predictor, output_scores_per_object

    # write the output masks as palette PNG files to output_mask_dir_under_video
    detector = MaskDetection(base_path=video_dataset_path, missing_threshold=missing_threshold, mixing_threshold=mixing_threshold)
    detect_results = {}
    for frame_idx, per_obj_output_mask in tqdm.tqdm(video_segments.items(), desc="Detecting failure"):
        per_obj_output_mask = {object_id: per_obj_mask_opening(per_obj_mask = per_obj_output_mask[object_id].squeeze().astype(np.uint8),
                                                               kernel_size = opening_kernel_size,
                                                               recover_threshold = opening_recover_threshold,
                                                               filter_threshold = filter_threshold) for object_id in per_obj_output_mask}

        masks_mixed, masks_missing, max_intersection, combined_mask = detector.detect(object_number,per_obj_output_mask)
        # store detection results for this frame
        detect_results[frame_idx] = {
            "masks_mixed": masks_mixed,
            "masks_missing": masks_missing,
            "max_intersection": max_intersection,
            "per_obj_output_mask": per_obj_output_mask,
            "combined_mask": combined_mask,
            "rgb_path": os.path.join(video_dataset_path, 'vos_rgb', batch_id, f'{frame_names[frame_idx]}.jpg')
        }

    save_till_index = len(detect_results)
    for frame_idx in detect_results:
        if detect_results[frame_idx]["masks_mixed"] or detect_results[frame_idx]["masks_missing"]:
            save_till_index = img_slider_viewer(detect_results=detect_results, initial_index=frame_idx)
            break

    for frame_idx in range(save_till_index):
        results = detect_results[frame_idx]
        ### save the mask to the output mask dir
        save_masks_to_dir(
            video_dataset_path=video_dataset_path,
            output_mask_dir_under_video=output_mask_dir_under_video,
            frame_name=frame_names[frame_idx],
            per_obj_output_mask=results["per_obj_output_mask"],
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=output_palette,
        )
        ### save the last mask of the batch to the input mask dir
        if frame_idx == list(video_segments.keys())[-1]:
            save_masks_to_dir(
                video_dataset_path=video_dataset_path,
                output_mask_dir_under_video=input_mask_dir_under_video,
                frame_name=frame_names[frame_idx],
                per_obj_output_mask=results["per_obj_output_mask"],
                height=height,
                width=width,
                per_obj_png_file=per_obj_png_file,
                output_palette=output_palette,
            )
    
    torch.cuda.empty_cache()
    del video_segments, detect_results
    return int(frame_names[save_till_index - 1]) if save_till_index > 0 else int(frame_names[0]) - 1  # Assume images in batches are continuous

def split_images_into_batches(image_folder: str, batch_size: int):
    """
    Split the images in the specified folder into different batches in ascending order of the file name numbers.
    Besides, except for the first batch, the first image of each batch is the same as the last image of the previous batch.

    Parameters:
    image_folder (str): The path of the folder where the images are stored.
    batch_size (int): The number of images contained in each batch.

    Returns:
    list: A list containing the paths of each batch folder, and the corresponding batch of images is stored in each batch folder.
    """
    # Get the file names of all jpg images in the image folder and sort them in ascending order according to the numbers in the file names.
    image_files = sorted([f for f in os.listdir(image_folder) if (f.endswith('.png') or f.endswith('.jpg'))], key=lambda x: int(x[:-4]))
    if len(image_files) == 0:
        return [batch_folder_path for batch_folder_path in os.listdir(image_folder) if os.path.isdir(batch_folder_path)]
    first_image_index = int(image_files[0][:-4])

    # Calculate the number of batches.
    num_batches = math.ceil((max([int(f[:-4]) for f in image_files]) + 1) / batch_size)

    batch_folders = []
    for batch_index in range(num_batches):
        # Create batch folders, such as batch_1, batch_2, etc.
        batch_folder_name = f"batch_{batch_index + 1}"
        batch_folder_path = os.path.join(image_folder, batch_folder_name)
        os.makedirs(batch_folder_path, exist_ok=True)
        batch_folders.append(batch_folder_path)

        start_index = batch_index * batch_size
        end_index = (batch_index + 1) * batch_size

        if batch_index > 0:
            # Get the path of the previous batch folder and the file name of the last image.
            prev_batch_folder = batch_folders[batch_index - 1]
            if len(os.listdir(prev_batch_folder)) > 0:
                prev_batch_last_image = sorted([f for f in os.listdir(prev_batch_folder) if (f.endswith('.png') or f.endswith('.jpg'))], key=lambda x: int(x[:-4]))[-1]
                prev_batch_last_image_path = os.path.join(prev_batch_folder, prev_batch_last_image)

                # Copy the last image of the previous batch to the current batch folder as the first image.
                shutil.copy(prev_batch_last_image_path, batch_folder_path)

        for i in range(start_index, end_index):
            # Move the images to the corresponding batch folders.
            if 0 <= (i - first_image_index) < len(image_files):
                source_file_path = os.path.join(image_folder, image_files[i - first_image_index])
                target_file_path = os.path.join(batch_folder_path, image_files[i - first_image_index])
                shutil.move(source_file_path, target_file_path)

    return batch_folders

def remove_data_till_images(image_folder: str, batch_size: int, remove_data_till: int):
    """
    Remove the images from the specified folder until (not including) the specified number of images.
    """
    # If the last image of the last batch is the last image to be removed, we need to remove the last batch.
    special_case = remove_data_till % batch_size == batch_size - 1

    remove_till_batch = remove_data_till // batch_size
    for batch in range(remove_till_batch + (1 if special_case else 0)):
        batch_folder = os.path.join(image_folder, f"batch_{batch + 1}")
        if os.path.exists(batch_folder):
            shutil.rmtree(batch_folder)
    if special_case:
        return

    # Process the remaining images
    remaining_batch_path = os.path.join(image_folder, f"batch_{remove_till_batch + 1}")
    remaining_images = sorted([f for f in os.listdir(remaining_batch_path) if (f.endswith('.png') or f.endswith('.jpg'))], key=lambda x: int(x[:-4]))
    for image in remaining_images:
        image_index = int(image[:-4])
        if image_index < remove_data_till:
            os.remove(os.path.join(remaining_batch_path, image))

def main(video_name: str,
         split: bool,
         batch_size: int = 400,
         remove_data_till: int = 0, external_args: Optional[argparse.Namespace] = None) -> int:

    USE_VOS_OPTIMIZED_VIDEO_PREDICTOR = False
    APPLY_POSTPROCESSING = True
    PER_OBJ_PNG_FILE = False
    USE_ALL_MASKS = False
    SCORE_THRESH: float = 0.0

    if external_args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--sam2_cfg",
            type=str,
            default="configs/sam2.1/sam2.1_hiera_l.yaml",
            help="SAM 2 model configuration file",
        )
        parser.add_argument(
            "--sam2_checkpoint",
            type=str,
            default="/data/recover/sam2/checkpoints/sam2.1_hiera_large.pt",
            help="path to the SAM 2 model checkpoint",
        )
        parser.add_argument(
            "--base_path",
            type=str,
            help="directory containing videos (as JPEG files) to run VOS prediction on",
            default="/data/DAVIS/JPEGImages/mice/"
        )
        parser.add_argument(
            "--input_mask_dir_under_video",
            type=str,
            help="RELATIVE PATH TO BASE_PATH/VIDEO_NAME: directory containing input masks (as PNG files) of each video",
            default="annotations"
        )
        parser.add_argument(
            "--output_mask_dir_under_video",
            type=str,
            help="RELATIVE PATH TO BASE_PATH/VIDEO_NAME: directory to save the output masks (as PNG files)",
            default="mask",
        )
        parser.add_argument(
            "--missing_threshold",
            type=int,
            default=500,
            help="the missing threshold for mask detection",
        )
        parser.add_argument(
            "--mixing_threshold",
            type=int,
            default=300,
            help="the mixing threshold for mask detection",
        )
        parser.add_argument(
            "--opening_kernel_size",
            default = (7 ,7),
            help="the recover kernel size after morphological opening",
        )
        parser.add_argument(
            "--opening_recover_threshold",
            default = 40,
            help="the recover threshold after morphological opening",
        )
        parser.add_argument(
            "--filter_threshold",
            default = 10,
            help="the filter threshold after morphological opening",
        )
        parser.add_argument(
            "--object_number",
            type=int,
            default=2,
            help="the number of objects to track in the video (default: 2, i.e. two objects tracking)",
        )
        '''
        parser.add_argument(
            "--score_thresh",
            type=float,
            default=0.0,
            help="threshold for the output mask logits (default: 0.0)",
        )
        parser.add_argument(
            "--use_all_masks",
            action="store_true",
            help="whether to use all available PNG files in input_mask_dir_under_video "
            "(default without this flag: just the first PNG file as input to the SAM 2 model; "
            "usually we don't need this flag, since semi-supervised VOS evaluation usually takes input from the first frame only)",
        )
        parser.add_argument(
            "--per_obj_png_file",
            action="store_true",
            help="whether use separate per-object PNG files for input and output masks "
            "(default without this flag: all object masks are packed into a single PNG file on each frame following DAVIS format; "
            "note that the SA-V dataset stores each object mask as an individual PNG file and requires this flag)",
        )
        parser.add_argument(
            "--apply_postprocessing",
            action="store_true",
            help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
            "(we don't apply such post-processing in the SAM 2 model evaluation)",
        )
        parser.add_argument(
            "--use_vos_optimized_video_predictor",
            action="store_true",
            help="whether to use vos optimized video predictor with all modules compiled",
        )
        '''
        args = parser.parse_args()
    else:
        args = external_args

    print(f"running VOS prediction on {video_name}")
    video_dataset_path = os.path.join(args.base_path, video_name)
    vos_rgb_path = os.path.join(video_dataset_path, "vos_rgb")


    # if we use per-object PNG files, they could possibly overlap in inputs and outputs
    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if PER_OBJ_PNG_FILE else "true")
    ]

    if USE_ALL_MASKS:
        print("using all available masks in input_mask_dir_under_video as input to the SAM 2 model")
    else:
        print("using only the first frame's mask in input_mask_dir_under_video as input to the SAM 2 model")

    # split the images into batches
    if split:
        split_images_into_batches(vos_rgb_path, batch_size)
    if remove_data_till > 0:
        remove_data_till_images(vos_rgb_path, batch_size, remove_data_till)

    batches = sorted(os.listdir(vos_rgb_path), key=lambda x: int(x.split('_')[-1]))

    last_successful_frame_number = 0
    for batch in batches:
        predictor = build_sam2_video_predictor(
            config_file=args.sam2_cfg,
            ckpt_path=args.sam2_checkpoint,
            apply_postprocessing=APPLY_POSTPROCESSING,
            hydra_overrides_extra=hydra_overrides_extra,
            vos_optimized=USE_VOS_OPTIMIZED_VIDEO_PREDICTOR,
        )
        last_successful_frame_number = vos_separate_inference_per_object(
            predictor=predictor,
            video_dataset_path=video_dataset_path,
            input_mask_dir_under_video=args.input_mask_dir_under_video,
            output_mask_dir_under_video=args.output_mask_dir_under_video,
            batch_id=batch,
            score_thresh=SCORE_THRESH,
            use_all_masks=USE_ALL_MASKS,
            per_obj_png_file=PER_OBJ_PNG_FILE,
            opening_kernel_size=args.opening_kernel_size,
            opening_recover_threshold=args.opening_recover_threshold,
            filter_threshold=args.filter_threshold,
            missing_threshold=args.missing_threshold,
            mixing_threshold=args.mixing_threshold,
            object_number=args.object_number
        )
        import gc
        gc.collect()

        batch_last_frame_number = max([int(frame_name.replace('.png', '').replace('.jpg', ''))
                                       for frame_name in os.listdir(os.path.join(vos_rgb_path, batch))])
        if last_successful_frame_number != batch_last_frame_number:
            print(f"last successful frame number: {last_successful_frame_number}, batch last frame number: {batch_last_frame_number}")
            break

    return last_successful_frame_number


if __name__ == "__main__":
    last_successful_frame_number = main("play_device0", False, remove_data_till=500)
    print(f"last successful frame number: {last_successful_frame_number}")
