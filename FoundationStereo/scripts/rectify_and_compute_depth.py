import cv2
import os
import sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import argparse
import numpy as np
import torch
import shutil
import tqdm
import sys
from omegaconf import OmegaConf
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.foundation_stereo import *



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stereo_parameters",
                       required=True,
                        help='path of json file storing the cameras parameters', )
    parser.add_argument("--base_path",
                        required=True,
                        help='the base path of the dataset',)
    parser.add_argument("--num_devices",
                        required=True,
                        type=int,
                        help='the number of devices', )
    parser.add_argument("--rectify",
                        # required=True,
                        action="store_true",
                        help="whether rectify the images", )
    parser.add_argument("--image_size",
                        default=(1440,1080),
                        help="image size of pictures", )
    parser.add_argument('--ckpt_path',
                        required=True,
                        type=str,
                        help='pretrained model path')
    parser.add_argument('--hiera',
                        default=0, type=int,
                        help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--valid_iters', type=int,
                        default=22,
                        help='number of flow-field updates during forward pass')
    parser.add_argument('--remove_invisible',
                        default=1, type=int,
                        help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')


    args = parser.parse_args()
    return args

def set_seed(random_seed):
  import torch,random
  np.random.seed(random_seed)
  random.seed(random_seed)
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def imread_unicode(path):
    with open(path, 'rb') as f:
        img_bytes = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    return img

def build_maps(device_parameters, device, camera, image_size):
    if camera == "left_camera":
        cameraMatrix = np.array(device_parameters[device]['left_camera']['camera_matrix'])
        distCoeffs = np.array(device_parameters[device]['left_camera']['dist_coeffs'])
        R = np.array(device_parameters[device]['stereo_params']['R_l'])
        P = np.array(device_parameters[device]['stereo_params']['intrinsic'])
    elif camera == "right_camera":
        cameraMatrix = np.array(device_parameters[device]['right_camera']['camera_matrix'])
        distCoeffs = np.array(device_parameters[device]['right_camera']['dist_coeffs'])
        R = np.array(device_parameters[device]['stereo_params']['R_r'])
        P = np.array(device_parameters[device]['stereo_params']['intrinsic'])
    map_x, map_y = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, P, image_size, cv2.CV_32FC1)
    return map_x, map_y

def rectify(path, map_x, map_y):
    img = imread_unicode(path)
    rectified_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return rectified_img

if __name__=="__main__":
    args = parse_args()
    print(args)
    device_parameters = json.load(open(args.stereo_parameters, 'r', encoding='utf-8-sig'))
    # rectify images
    if args.rectify:
        for device_idx in range(args.num_devices):
            device = f"device{device_idx}"
            for cam in ["left_camera","right_camera"]:
                src_folder = os.path.join(args.base_path, device, cam)
                dst_folder = os.path.join(args.base_path, device, cam+"_rectified")
                os.makedirs(dst_folder, exist_ok=True)

                map_x, map_y = build_maps(device_parameters, device,cam, args.image_size)
                images = sorted(
                    [img for img in os.listdir(src_folder) if img.lower().endswith(('.jpg', '.png', '.jpeg'))])
                for image in images:
                    rectified_img = rectify(os.path.join(src_folder,image),map_x,map_y)
                    frame_id = image.split('_')[-1].split('.')[0]
                    cv2.imwrite((f"{dst_folder}/{frame_id}.jpg"), rectified_img)

    # compute depth using Foundation Stereo
    if not args.rectify:

        set_seed(0)
        torch.autograd.set_grad_enabled(False)
        ckpt_path = args.ckpt_path
        cfg = OmegaConf.load(f'{os.path.dirname(ckpt_path)}/cfg.yaml')
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        for k in args.__dict__:
            cfg[k] = args.__dict__[k]
        args = OmegaConf.create(cfg)
        model = FoundationStereo(args)

        ckpt = torch.load(ckpt_path,weights_only=False)
        model.load_state_dict(ckpt['model'])

        model.cuda()
        model.eval()

        for device_idx in range(args.num_devices):
            device = f"device{device_idx}"
            params = device_parameters[device]
            base_path = os.path.join(args.base_path, device)
            depth_path = os.path.join(args.base_path, device, 'depth')
            os.makedirs(depth_path, exist_ok=True)
            vis_depth_path = os.path.join(args.base_path, device, 'vis_depth')
            os.makedirs(vis_depth_path, exist_ok=True)
            left_dir = os.path.join(base_path, 'left_camera_rectified')
            right_dir = os.path.join(base_path, 'right_camera_rectified')

            left_images = sorted(
                [img for img in os.listdir(left_dir) if img.lower().endswith(('.jpg', '.png', '.jpeg'))])
            right_images = sorted(
                [img for img in os.listdir(right_dir) if img.lower().endswith(('.jpg', '.png', '.jpeg'))])
            if left_images != right_images:
                raise ValueError(f"The file names of the left and right camera images are inconsistent！{base_path}")

            for image in left_images:
                filename = image.split('.')[0]

                img0 = imageio.imread(os.path.join(left_dir, f"{image}"))
                img1 = imageio.imread(os.path.join(right_dir, f"{image}"))
                H, W = img0.shape[:2]
                img0_ori = img0.copy()
                img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
                img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
                padder = InputPadder(img0.shape, divis_by=32, force_square=False)
                img0, img1 = padder.pad(img0, img1)

                with torch.cuda.amp.autocast(True):
                    if not args.hiera:
                        disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
                    else:
                        disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
                disp = padder.unpad(disp.float())
                disp = disp.data.cpu().numpy().reshape(H, W)

                if args.remove_invisible:
                    yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
                    us_right = xx - disp
                    invalid = us_right < 0
                    disp[invalid] = np.inf

                K = np.array(params["stereo_params"]["intrinsic"])
                baseline = float(params["stereo_params"]["baseline"][0])
                depth = K[0, 0] * baseline / (disp*1000)
                # save depth
                depth = (depth*1000).astype(np.uint16)  # *1000
                cv2.imwrite(f'{depth_path}/{filename}.png', depth)

                vis = vis_disparity(disp)
                vis = np.concatenate([img0_ori, vis], axis=1)
                imageio.imwrite(f'{vis_depth_path}/{filename}.png', vis)