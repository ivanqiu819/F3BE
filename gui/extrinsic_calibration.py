
import cv2
import json
import aprilgrid
import numpy as np
import sys
from typing import List
import os




def calibrate(rgb_imgs: List[np.ndarray], intrinsics: List[np.ndarray], april_grid_size, april_size, april_interval, AprilTag_family) -> List[np.ndarray]:
    """Calculate the extrinsic parameters of each camera relative to camera 0 using N images of the same AprilGrid taken by different cameras.

    Args:
        rgb_imgs (List[np.ndarray]): N RGB images taken by N cameras
        intrinsics (List[np.ndarray]): Intrinsic matrices of each camera

    Returns:
        List[np.ndarray]: Extrinsic matrices of each camera relative to camera 0
    """
    COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 255, 255),
    ]
    poses_to_world: List[np.ndarray] = []
    detector = aprilgrid.Detector(AprilTag_family)
    for cam_id in range(len(rgb_imgs)):
        vis = rgb_imgs[cam_id].copy()
        gray = cv2.cvtColor(rgb_imgs[cam_id], cv2.COLOR_BGR2GRAY)

        tags = detector.detect(gray)
        if len(tags) == 0:
            raise ValueError(f"No AprilGrid tags detected for camera {cam_id}")

        # Extract corner points of each marker
        world_points = []
        image_points = []
        for marker in tags:
            marker_id = marker.tag_id
            row = marker_id // april_grid_size
            col = marker_id % april_grid_size
            
            # Calculate the actual coordinates of the marker
            marker_lower_left_world = np.array([
                col * (april_size + april_interval),
                row * (april_size + april_interval),
                0
            ])

            corners = marker.corners.reshape(4, 2)
            swp = corners[2, :].copy()
            corners[2, :] = corners[3, :]
            corners[3, :] = swp
            for i, corner in enumerate(corners):
                cv2.circle(vis, tuple(corner.round().astype(int)), 3, COLORS[i], -1)

                corner_world = marker_lower_left_world + np.array([
                    (i % 2) * april_size,
                    (i // 2) * april_size,
                    0
                ])

                world_points.append(corner_world)
                image_points.append(corner)



        world_points = np.asarray(world_points).astype(float)
        image_points = np.asarray(image_points).astype(float)
        print(f'Using {len(world_points)} points for camera {cam_id}')

        ret, rvec, tvec = cv2.solvePnP(world_points, image_points, intrinsics[cam_id], np.array([]))
        assert ret, f"Failed to solve PnP for camera {cam_id}"

        # Calculate the average residual
        projected_points, _ = cv2.projectPoints(world_points, rvec, tvec, intrinsics[cam_id], np.array([]))
        projected_points = projected_points[:, 0, :]
        residual = np.linalg.norm(projected_points - image_points, axis=1).mean()
        print(f'Average residual of camera {cam_id}: {residual:.3f} pixels')

        # Construct the pose matrix
        mat = np.eye(4)
        mat[:3, :3] = cv2.Rodrigues(rvec)[0]
        mat[:3, 3:] = tvec
        poses_to_world.append(mat)

    poses_to_zero = []
    for i in range(len(poses_to_world)):
        poses_to_zero.append(poses_to_world[i] @ np.linalg.inv(poses_to_world[0]))

    return poses_to_zero

def imread_unicode(path):
    with open(path, 'rb') as f:
        img_bytes = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    return img

def build_maps(device_parameters, device, image_size):
    cameraMatrix = np.array(device_parameters[device]['left_camera']['camera_matrix'])
    distCoeffs = np.array(device_parameters[device]['left_camera']['dist_coeffs'])
    R = np.array(device_parameters[device]['stereo_params']['R_l'])
    P = np.array(device_parameters[device]['stereo_params']['intrinsic'])
    map_x, map_y = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, P, image_size, cv2.CV_32FC1)
    return map_x, map_y

def rectify(path, map_x, map_y):
    img = imread_unicode(path)
    rectified_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return rectified_img

def main(project_path):
    checker_path = os.path.dirname(project_path)
    proj_path = os.path.dirname(checker_path)
    config_path = os.path.join(proj_path, "config.json")
    config_data = json.load(open(config_path, "r",encoding='utf-8-sig'))
    camera_para_path = os.path.join(proj_path, "stereo_parameter.json")
    camera_paras = json.load(open(camera_para_path, "r",encoding='utf-8-sig'))

    camera_num = config_data["Project_Info"]["Number_of_Devices"]
    april_size = config_data["Calibration_Info"]["Extrinsic"]["April_Size"]
    april_interval = config_data["Calibration_Info"]["Extrinsic"]["April_Interval"]
    april_grid_size = config_data["Calibration_Info"]["Extrinsic"]["April_Grid_Size"]
    AprilTag_family = config_data["Calibration_Info"]["Extrinsic"]["April_Family"]
    rgb_list = []
    intrinsic_devices = []
    image_size = (1440,1080)

    for camera_idx in range (camera_num):
        left_path = os.path.join(project_path, f"device{camera_idx}")
        left_files = os.listdir(left_path)
        rgb = cv2.imread(f'{left_path}/{left_files[0]}')
        # rgb_list.append(rgb)
        device = f"device{camera_idx}"
        map_x, map_y = build_maps(camera_paras, device, image_size)
        rectified_img = rectify(os.path.join(left_path, left_files[0]), map_x, map_y)
        rgb_list.append(rectified_img)
        intrinsic = np.array(camera_paras["device{}".format(camera_idx)]["stereo_params"]["intrinsic"])
        intrinsic_devices.append(intrinsic)
    poses_wrt_world = calibrate(rgb_list, intrinsic_devices,april_grid_size,april_size,april_interval,AprilTag_family)

    for source, pose in enumerate(poses_wrt_world):
        camera_paras["device{}".format(source)][f"pose_wrt_world"] = pose.tolist()
        for reference in range(camera_num):
            if source == reference:
                continue
            source_wrt_reference = (np.linalg.inv(poses_wrt_world[reference]) @ pose)
            camera_paras["device{}".format(source)][f"device{source}_wrt_device{reference}"] = source_wrt_reference.tolist()
    with open(camera_para_path, 'w', encoding='utf-8-sig') as f:
        json.dump(camera_paras, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python extrinsic_calibration.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    main(file_path)
