import os
import shutil
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import cv2
import matplotlib
matplotlib.use('TkAgg')

from typing import Dict

def put_per_obj_mask(per_obj_mask: Dict[int, np.ndarray], height: int, width: int) -> np.ndarray:
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id].astype(bool)
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask

def img_slider_viewer(detect_results, initial_index: int):
    """
    Visualize RGB and Mask image pairs, switch images via slider, and annotate yes/no with buttons.
    Returns annotation result dictionary.
    """
    num_images = len(detect_results)

    # Load the i-th image
    def load_image(i):
        processed_mask = detect_results[i]['combined_mask']
        height, width = processed_mask.shape[0], processed_mask.shape[1]

        original_mask = put_per_obj_mask(detect_results[i]['per_obj_output_mask'], height, width)
        rgb = cv2.cvtColor(cv2.imread(detect_results[i]['rgb_path']), cv2.COLOR_BGR2RGB)
        masks_mixed = detect_results[i]['masks_mixed']
        masks_missing = detect_results[i]['masks_missing']
        max_intersection = detect_results[i]['max_intersection']
        return rgb, original_mask, processed_mask, masks_mixed, masks_missing, max_intersection

    # Initialize images
    rgb_img, original_mask, processed_mask, masks_mixed, masks_missing, max_intersection = load_image(initial_index)

    # Create image window
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.35)
    rgb_ax = axes[0].imshow(rgb_img)
    axes[0].set_title("RGB")
    original_mask_ax = axes[1].imshow(original_mask)
    axes[1].set_title("Original Mask")
    processed_mask_ax = axes[2].imshow(processed_mask)
    axes[2].set_title("Processed Mask")

    # Place slider at the bottom
    ax_slider = plt.axes([0.2, 0.2, 0.6, 0.03])
    slider = Slider(ax_slider, 'Image Index', 0, num_images - 1, valinit=initial_index, valstep=1)

    # Button area
    ax_button_yes = plt.axes([0.25, 0.1, 0.1, 0.05])
    ax_button_no = plt.axes([0.65, 0.1, 0.1, 0.05])
    button_yes = Button(ax_button_yes, 'Yes')
    button_no = Button(ax_button_no, 'No')

    # Record yes/no button press information
    annotations = {}

    # Add text on mask image (initialization)
    mask_text = axes[1].text(
        0.5, 0.05, f"Index: {initial_index}", color='red', fontsize=14,
        ha='center', va='bottom', transform=axes[1].transAxes
    )
    # Add text on processed mask image (initialization)
    processed_mask_text = axes[2].text(
        0.5, 0.05,
        f"mixed: {masks_mixed}, missing: {masks_missing}, max_intersection: {max_intersection}",
        color='blue', fontsize=12, ha='center', va='bottom', transform=axes[2].transAxes
    )

    # --- Help area and buttons placed at the bottom ---
    ax_help = plt.axes([0.05, 0.01, 0.75, 0.08])  # Near the bottom
    ax_help.axis('off')
    help_text = (
        "Help: Use left/right arrow keys to switch images; or drag the slider to select an image; \n"
        "Yes: the frame where failure starts (frames before are correct). \n"
        "No: the whole batch video is correct. \n"
        "Red lines: frames where masks_mixed or masks_missing is True."
    )
    help_text_obj = ax_help.text(0.5, 0.5, help_text, ha='center', va='center', fontsize=12, color='black')
    ax_help.set_visible(False)  # Default hidden

    ax_help_btn = plt.axes([0.82, 0.01, 0.13, 0.08])  # Adjacent to help area
    btn_help = Button(ax_help_btn, 'Help')

    def on_help_clicked(event):
        ax_help.set_visible(not ax_help.get_visible())
        fig.canvas.draw_idle()

    btn_help.on_clicked(on_help_clicked)

    # Collect indices that need highlighting
    highlight_indices = [i for i, d in enumerate(detect_results) if detect_results[d]['masks_mixed'] or detect_results[d]['masks_missing']]
    for idx in highlight_indices:
        ax_slider.axvline(idx, color='red', ymin=0, ymax=1, linewidth=2)

    # Image update callback
    def update(val):
        idx = int(slider.val)
        rgb_img, original_mask, processed_mask, masks_mixed, masks_missing, max_intersection = load_image(idx)
        rgb_ax.set_data(rgb_img)
        original_mask_ax.set_data(original_mask)
        processed_mask_ax.set_data(processed_mask)
        # Update text on mask
        mask_text.set_text(f"Index: {idx}")
        # Update text on processed mask
        processed_mask_text.set_text(
            f"mixed: {masks_mixed}, missing: {masks_missing}, max_iou: {max_intersection:.2f}"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Button click handling
    result = None  # Used to store the final returned index

    def on_yes_clicked(event):
        idx = int(slider.val)
        annotations[idx] = 'yes'
        print(f"Image {idx}: YES")
        nonlocal result
        result = idx
        plt.close(fig)  # Close window

    def on_no_clicked(event):
        idx = int(slider.val)
        annotations[idx] = 'no'
        print(f"Image {idx}: NO")
        nonlocal result
        result = num_images
        plt.close(fig)  # Close window

    button_yes.on_clicked(on_yes_clicked)
    button_no.on_clicked(on_no_clicked)

    # Keyboard events: left/right arrow keys control slider
    def on_key(event):
        idx = int(slider.val)
        if event.key == 'right':
            if idx < num_images - 1:
                slider.set_val(idx + 1)
        elif event.key == 'left':
            if idx > 0:
                slider.set_val(idx - 1)

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show(block=True)
    plt.close(fig)

    return result

def add_depth_gradient_to_rgb(idx, depth_path, rgb_path, save_path, threshold, kernel_size, scale):
    if scale != 0:
        # Read the depth map
        depth_map = cv2.imread(f"{depth_path}/{idx}.png", cv2.IMREAD_UNCHANGED) / 1000

        # Read the RGB image (support both .jpg and .png)
        rgb_image = None
        for ext in ['jpg', 'png']:
            rgb_file = f'{rgb_path}/{idx}.{ext}'
            if os.path.exists(rgb_file):
                rgb_image = cv2.imread(rgb_file)
                break
        if rgb_image is None:
            raise FileNotFoundError(f'RGB image not found for {idx} in {rgb_path} (tried .jpg and .png)')
        # Get depth gradient
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=kernel_size)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=kernel_size)
        gradient_map = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Clip according to threshold
        gradient_map[gradient_map < threshold[0]] = 0
        gradient_map = np.clip(gradient_map, 0, threshold[1])

        gradient_map = cv2.normalize(gradient_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')


        # add gradient map to RGB image
        # Expand gradient_map to 3 channels to match rgb_image shape
        gradient_map_3ch = cv2.merge([gradient_map, gradient_map, gradient_map])
        rgb_image_with_gradient = rgb_image.astype('float32') + gradient_map_3ch.astype('float32') * scale  # Adjust the scaling factor as needed
        rgb_image_with_gradient = np.clip(rgb_image_with_gradient, 0, 255).astype('uint8')
    if scale == 0:
        # Read the RGB image (support both .jpg and .png)
        rgb_image = None
        for ext in ['jpg', 'png']:
            rgb_file = f'{rgb_path}/{idx}.{ext}'
            if os.path.exists(rgb_file):
                rgb_image_with_gradient = cv2.imread(rgb_file)
                break
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(f'{save_path}/{idx}.jpg', rgb_image_with_gradient)


def add_depth_gradient_worker(args):
    idx, depth_path, rgb_path, save_path, threshold, kernel_size, scale = args
    add_depth_gradient_to_rgb(
        idx,
        depth_path=depth_path,
        rgb_path=rgb_path,
        save_path=save_path,
        threshold=threshold,
        kernel_size=kernel_size,
        scale=scale
    )