import cv2
import os
import numpy as np
import argparse
import json
import sys

def update_nested(data, key_path, value):
    keys = key_path.split('.')
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stereo_calibration.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    checker_path = os.path.dirname(file_path)
    proj_path = os.path.dirname(checker_path)
    config_path = os.path.join(proj_path, "config.json")
    camera_para_path = os.path.join(proj_path, "stereo_parameter.json")
    camera_paras = {}

    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config_data = json.load(f)
    cam_num = config_data["Project_Info"]["Number_of_Devices"]
    chessboard = config_data['Calibration_Info']["Intrinsic"]["Inner_Corner_Array_Size"]
    square_size = config_data['Calibration_Info']["Intrinsic"]["Checkerboard_Corner_Spacing"]
    updates_sum = {}
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((1, chessboard[0] * chessboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
    objp[0, :, 0] *= square_size[0]
    objp[0, :, 1] *= square_size[1]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((1, chessboard[0]*chessboard[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
    objp[0,:,0] *= square_size[0]
    objp[0,:,1] *= square_size[1]

    for cam_idx in range(cam_num):
        imgpoints_l = []
        imgpoints_r = []
        objpoints = []
        left_path = os.path.join(file_path, f"device{cam_idx}","left")
        right_path = os.path.join(file_path, f"device{cam_idx}","right")
        left_files = sorted(os.listdir(left_path))
        right_files = sorted(os.listdir(right_path))
        num = len(left_files)
        for j in range(num):
            img_l = cv2.imread(f"{left_path}/{left_files[j]}")
            gray_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
            img_r = cv2.imread(f"{right_path}/{right_files[j]}")
            gray_r = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard)
            if ret_l and ret_r:
                objpoints.append(objp)
                corners2_l = cv2.cornerSubPix(gray_l,corners_l,(11,11),(-1,-1),criteria)
                imgpoints_l.append(corners2_l)
                corners2_r = cv2.cornerSubPix(gray_r,corners_r,(11,11),(-1,-1),criteria)
                imgpoints_r.append(corners2_r)
        print(f'Single camera calibration error of device{cam_idx}')
        ret, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1],None,None)
        print(f'    ret of left camera: ', ret)
        ret, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1],None,None)
        print(f'    ret of right camera: ', ret)

        # stereo calibration
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
            cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1])
        baseline = np.sqrt(T[0] ** 2 + T[1] ** 2 + T[2] ** 2) # baseline calculation

        print(f"Stereo Calibration Error of device{cam_idx} :")
        print(f"    Camera matrix ret :",retval)
        print('\n')

        (R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2) = \
                    cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, gray_r.shape[::-1], R,
                                      T,flags = cv2.CALIB_ZERO_DISPARITY)

        camera_paras[f'device{cam_idx}'] = {
            'left_camera': {
                'camera_matrix': cameraMatrix1.tolist(),
                'dist_coeffs': distCoeffs1.tolist()
            },
            'right_camera': {
                'camera_matrix': cameraMatrix2.tolist(),
                'dist_coeffs': distCoeffs2.tolist()
            },
            'stereo_params': {
                'R_l': R_l.tolist(),
                'R_r': R_r.tolist(),
                'T': T.T[0].tolist(),
                'intrinsic': P_l[:, :-1].tolist(),
                'baseline': baseline.tolist()

            }
        }


        h, w = img_l.shape[:2]
        map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R_l, P_l, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R_r, P_r, (w, h), cv2.CV_32FC1)

        # stereo rectification
        rectified_l = cv2.remap(img_l, map1x, map1y, cv2.INTER_LINEAR)
        rectified_r = cv2.remap(img_r, map2x, map2y, cv2.INTER_LINEAR)

        height = h
        width = w+w
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:h, :w] = rectified_l
        canvas[:h, w:w+w] = rectified_r

        # Draw thin lines between the two images
        line_color = (0, 255, 0)  # Green line, BGR format
        line_thickness = 1  # Line thickness
        for i in range(10):
            dis = int(h/10*(i+1))
            cv2.line(canvas, (0, dis), (2*w, dis), line_color, line_thickness)

        # Save results
        save_path = os.path.join(file_path, f"device{cam_idx}")
        cv2.imwrite(f'{save_path}/rectified_stereo_image.jpg', canvas)

    json_str = json.dumps(camera_paras, indent=4, ensure_ascii=False)
    with open(camera_para_path, 'w', encoding='utf-8-sig') as f:
        f.write(json_str)
