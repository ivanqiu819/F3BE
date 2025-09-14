import shutil
import numpy as np
import cv2
import os
from PIL import Image
import re
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from typing import Dict, Tuple

# TODO: 让可视化颜色一致
class MaskDetection:
    def __init__(
                self,
                base_path: str,
                missing_threshold: str,
                mixing_threshold: str):
        self.base_path = base_path
        self.missing_threshold = missing_threshold
        self.mixing_threshold = mixing_threshold

    @staticmethod
    def load_ann_png(path):
        """
        Load a PNG file as a mask and its associated palette.

        Args:
            path (str): The path to the PNG file containing the mask.

        Returns:
            np.ndarray: The mask as a NumPy array.
            list: The palette used in the PNG image (not used in further processing here).
        """
        mask = Image.open(path)
        palette = mask.getpalette()
        mask = np.array(mask).astype(np.uint8)
        return mask, palette

    @staticmethod
    def generate_total_mask_from_sparse(mask):
        """
        Generate a total mask from a sparse mask by computing its convex hull.

        Args:
            mask (np.ndarray): A sparse binary mask (0s and 1s).

        Returns:
            np.ndarray: The expanded mask, where the convex hull is filled.
        """
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        non_zero_coords = np.argwhere(mask > 0)
        non_zero_coords[:, [0, 1]] = non_zero_coords[:, [1, 0]]

        if non_zero_coords.shape[0] == 0:
            return mask

        hull = cv2.convexHull(non_zero_coords)
        total_mask = np.zeros_like(mask, dtype=np.uint8)
        hull_points = hull.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(total_mask, [hull_points], 1)

        return total_mask

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

    def detect(self, object_number, per_obj_mask: Dict[int, np.ndarray]) -> Tuple[bool, bool, int, np.ndarray]:
        ### initialization
        intersection_color = [255, 255, 255]
        masks_mix = False
        masks_missing = False
        max_intersection = 0

        ### mask missing detection
        masks = list(per_obj_mask.values())
        if len(masks) < object_number:
            masks_missing = True
        for mask in masks:
            if np.count_nonzero(mask) < self.missing_threshold:
                masks_missing = True
                break

        ### mask mixing detection
        total_masks = [self.generate_total_mask_from_sparse(mask) for mask in masks]

        combined_mask = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)

        cmap = plt.get_cmap('viridis', len(total_masks))
        colors = [(int(255 * c[0]), int(255 * c[1]), int(255 * c[2])) for c in cmap(np.linspace(0, 1, len(total_masks)))]

        max_intersection = 0
        for i, total_mask in enumerate(total_masks):
            blue_mask = np.all(combined_mask == intersection_color, axis=-1)
            non_blue_total_mask = np.logical_and(total_mask == 1, ~blue_mask)
            combined_mask[non_blue_total_mask] = colors[i]

            for j in range(i + 1, len(total_masks)):
                intersection = np.logical_and(total_mask, total_masks[j])
                intersection_count = np.count_nonzero(intersection)
                if intersection_count > self.mixing_threshold and intersection_count > max_intersection:
                    masks_mix = True
                    max_intersection = intersection_count
                    combined_mask[intersection] = intersection_color

        return masks_mix, masks_missing, max_intersection, combined_mask

def main():
    parser = argparse.ArgumentParser(description='Process mask files.')
    parser.add_argument('--base_path', default='/data1/mice/0516/', help='Base directory for data')
    parser.add_argument('--filename', type=int, default=50, help='index number of the mask file to process')
    parser.add_argument('--missing_threshold', type=int, default=500, help='Minimum number of non-zero pixels for a valid mask (for mask missing check)')
    parser.add_argument('--mixing_threshold', type=int, default=300, help='Minimum number of intersection pixels for mask mixing check')
    parser.add_argument('--visualize', default= True, help='Show visualization of combined masks')

    args = parser.parse_args()

    mask_detection = MaskDetection(
        base_path=args.base_path,
        filename=args.filename,
        missing_threshold=args.missing_threshold,
        mixing_threshold=args.mixing_threshold,
        visualize=args.visualize)

    mask_detection.detect()


if __name__ == "__main__":
    main()
