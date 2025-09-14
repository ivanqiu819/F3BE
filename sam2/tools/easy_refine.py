import os
import json
import shutil
import argparse
import logging
import tqdm
from utils import add_depth_gradient_worker, add_depth_gradient_to_rgb
from refinement import Refinement
from vos_inference import main as vos_inference_main
from typing import Dict, List
from multiprocessing import Pool


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Easy Refine for SAM2')
    ### should the paths of depths(not added yet), masks, and rgb set separately or just set the base path?
    ### Refinement Parameters
    parser.add_argument('--base_path', type=str, required=True, help='base path containing the source device and reference devices')
    parser.add_argument('--device_number', type=int, required=True, help='the number of devices')
    parser.add_argument('--max_refine_frames', type=int, default=5, help='the maximum number of frames to be refined')
    parser.add_argument('--sam2_checkpoint', type=str, required=True, help='Path to the SAM2 checkpoint')
    parser.add_argument('--sam2_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml', help='Path to the SAM2 model configuration file')
    parser.add_argument('--parameter_file', type=str, required=True, help='Path to the JSON file containing device parameters')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the JSON file containing other parameters')
    parser.add_argument("--GPU_device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument('--visualize', default=False, action='store_true', help='Show visualization of refined masks')
    parser.add_argument('--depth_threshold', type=float, default=0.001, help='Depth threshold for mask refinement, if the depth difference is larger than this value, the point will be filtered out')
    parser.add_argument('--color_threshold', type=float, default=300, help='Color threshold for mask refinement, if the brightness of the pixel is larger than this value, the point will be filtered out')

    ### Detection Parameters
    parser.add_argument("--object_number", type=int, default=2, help="the number of objects in the frame, used for mask missing detection")
    parser.add_argument("--missing_threshold", type=int, default=1500, help="the missing threshold for mask detection")
    parser.add_argument("--mixing_threshold", type=int, default=2000, help="the mixing threshold for mask detection")

    ### Pre-processing Parameters
    parser.add_argument("--depth_gradient_threshold", type=float, default=[0.05, 0.7], help="the threshold for depth gradient, if the depth gradient is larger than this value, the point will be filtered out")
    parser.add_argument("--sobel_kernel_size", type=int, default=3, help="the kernel size for sobel operator, the larger the kernel size, the smoother the depth gradient will be")
    parser.add_argument("--depth_gradient_scale", type=int, default=4, help="the scale for depth gradient, the larger the scale, the more the depth gradient will be added to the RGB image")

    ### VOS Inference Parameters
    parser.add_argument("--input_mask_dir_under_video", type=str,
                        help="RELATIVE PATH TO BASE_PATH/VIDEO_NAME: directory containing input masks (as PNG files) of each video",
                        default="annotation")
    parser.add_argument("--output_mask_dir_under_video", type=str,
                        help="RELATIVE PATH TO BASE_PATH/VIDEO_NAME: directory containing input masks (as PNG files) of each video",
                        default="mask")
    parser.add_argument("--opening_kernel_size", default=(15, 15), help="the kernel size for morphological opening")
    parser.add_argument("--opening_recover_threshold", type=int, default=500, help="the recover threshold after morphological opening,for the pixels that are removed by morphological opening, if the number of pixels is smaller than this value, the pixel will be recovered")
    parser.add_argument("--filter_threshold", default = 800, help="the filter threshold after morphological opening and recover process, normally used to filter out the noise away from mask, if the number of pixels is smaller than this value, the pixel will be removed")
    args = parser.parse_args()

    device_parameters = json.load(open(args.parameter_file, encoding='utf-8-sig'))
    config_parameters = json.load(open(args.config_file, encoding='utf-8-sig'))
    progress: Dict[str, int] = {device: 0 for device in device_parameters.keys()}  # Included
    the_first_device: str = list(device_parameters.keys())[0]
    image_list = os.listdir(os.path.join(args.base_path, the_first_device, 'left_camera_rectified'))
    max_frame_index: int = max([int(image.split('.')[0]) for image in image_list if (image.endswith('.jpg') or image.endswith('.png'))])

    start_progress = {}
    for i in range(args.device_number):
        start_progress[f"device{i}"] = 0
    progress = start_progress.copy()

    # copy data to EasyRefine directory
    for device in progress.keys():
        vos_rgb_dir = os.path.join(args.base_path, device, 'vos_rgb')
        if not os.path.exists(vos_rgb_dir):
            image_dir = os.path.join(args.base_path, device, 'left_camera_rectified')
            image_list = [img for img in os.listdir(image_dir) if img.endswith('.jpg') or img.endswith('.png')]
            tasks = []
            for image in image_list:
                idx = os.path.basename(image).split('.')[0]
                tasks.append((
                    idx,
                    os.path.join(args.base_path, device, 'depth'),
                    image_dir,
                    vos_rgb_dir,
                    args.depth_gradient_threshold,
                    args.sobel_kernel_size,
                    args.depth_gradient_scale
                ))
            os.makedirs(vos_rgb_dir, exist_ok=True)
            with Pool(processes=os.cpu_count()) as pool:
                list(tqdm(pool.imap_unordered(add_depth_gradient_worker, tasks), total=len(tasks), desc=f"Copying images for {device}"))

    while not all([v == max_frame_index for v in progress.values()]):
        logging.info(f"Current progress: {progress}")
        taken_action: bool = False

        # Check whether there's a video sequence that can be refined
        for device in progress.keys():
            reference_devices: List[str] = config_parameters["Reference_Devices"][device]
            device_started = (progress[device] > start_progress[device])

            if device_started and all([progress[ref_device] > progress[device] for ref_device in reference_devices]):
                refinement = Refinement(
                    base_path=args.base_path,
                    source_device=device,
                    sam2_checkpoint=args.sam2_checkpoint,
                    sam2_cfg=args.sam2_cfg,
                    config_file=args.parameter_file,
                    visualize=args.visualize,
                    GPU_device=args.GPU_device,
                    depth_threshold=args.depth_threshold,
                    color_threshold=args.color_threshold
                )
                refine_interval = min([args.max_refine_frames] + [progress[ref_device] - progress[device] for ref_device in reference_devices])
                logging.info(f"Refining {device} from frame {progress[device] + 1} to {progress[device] + refine_interval}")

                # copy the cooresponding video sequence to the refinement directory
                for index in range(progress[device] + 1, progress[device] + refine_interval + 1):
                    add_depth_gradient_to_rgb("%06d" % index,
                                      depth_path=os.path.join(args.base_path, device, 'depth'),
                                      rgb_path=os.path.join(args.base_path, device, 'left_camera_rectified'),
                                      save_path=os.path.join(args.base_path, device, 'refine', f"{index:06d}"),
                                      threshold=args.depth_gradient_threshold,
                                      kernel_size=args.sobel_kernel_size,
                                      scale=args.depth_gradient_scale)
                # Refine the video sequence
                refinement.refine()
                # Copy refined masks to the mask directory
                for index in range(progress[device] + 1, progress[device] + refine_interval + 1):
                    shutil.copy(os.path.join(args.base_path, device, 'refine', "%06d" % index, "%06d.png" % index),
                                os.path.join(args.base_path, device, args.output_mask_dir_under_video, "%06d.png" % index))
                # Copy annotation
                annotation_index = progress[device] + refine_interval
                shutil.copy(os.path.join(args.base_path, device, 'refine', "%06d" % annotation_index, "%06d.png" % annotation_index),
                            os.path.join(args.base_path, device, args.input_mask_dir_under_video, "%06d.png" % annotation_index))
                # remove the refined data
                shutil.rmtree(os.path.join(args.base_path, device, 'refine'))
                progress[device] = vos_inference_main(device, False, remove_data_till=progress[device] + refine_interval, external_args=args)
                logging.info(f"{device} progressed to {progress[device]}")

                taken_action = True
                break
        if taken_action:
            continue

        # Check whether there's a video sequence that hasn't started
        for device in progress.keys():
            if progress[device] == start_progress[device] and progress[device] < max_frame_index:
                logging.info(f"Starting {device}")
                progress[device] = vos_inference_main(device, True, remove_data_till=progress[device], external_args=args)
                logging.info(f"{device} progressed to {progress[device]}")

                taken_action = True
                break
        if taken_action:
            continue

        raise Exception(f"No action can be taken, current progress: {progress}")
