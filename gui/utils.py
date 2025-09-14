import numpy as np
import cv2

def depth2pcd(depth, intrinsic, extrinsic, mask=None):
    """
    Convert depth map to point cloud.
    
    Args:
        depth (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (4, 4).
        mask (np.ndarray, optional): Mask to filter points. If None, all points are used.
        
    Returns:
        np.ndarray: Point cloud of shape (N, 3), where N is the number of valid points.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert depth to 3D points
    z = depth.flatten()
    x = (u.flatten() - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v.flatten() - intrinsic[1, 2]) * z / intrinsic[1, 1]
    
    points_3d = np.vstack((x, y, z)).T
    
    if mask is not None:
        mask_flat = mask.flatten()
        points_3d = points_3d[mask_flat > 0]
    
    # Apply extrinsic transformation
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_3d_transformed = points_3d_homogeneous @ extrinsic.T
    
    return points_3d_transformed[:, :3]  # Return only x, y, z coordinates