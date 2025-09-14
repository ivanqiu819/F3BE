from PIL import Image
import os
import re
import argparse
from tqdm import tqdm
from multiprocessing import Pool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default="/data2/0716", help="path contains videos")
    parser.add_argument("--video_list",
                        default=['fight2'],
                        help="list of video folders. eg. v1, v2, v3...", )
    parser.add_argument("--device_list", default=["0"], help="list of cameras", )
    args = parser.parse_args()
    return args

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def colorize_rgb_with_mask(rgb_image_path, mask_image_path, saving_path):
    try:
        rgb_image = Image.open(rgb_image_path).convert("RGB")
        mask_image = Image.open(mask_image_path).convert("P")
        palette = mask_image.getpalette()
        width, height = rgb_image.size
        colored_image = Image.new("RGB", (width, height))
        for y in range(height):
            for x in range(width):
                mask_pixel = mask_image.getpixel((x, y))
                if mask_pixel != 0:
                    r = palette[mask_pixel * 3]
                    g = palette[mask_pixel * 3 + 1]
                    b = palette[mask_pixel * 3 + 2]
                    original_r, original_g, original_b = rgb_image.getpixel((x, y))
                    alpha = 0.7
                    new_r = int(original_r * (1 - alpha) + r * alpha)
                    new_g = int(original_g * (1 - alpha) + g * alpha)
                    new_b = int(original_b * (1 - alpha) + b * alpha)
                    colored_image.putpixel((x, y), (new_r, new_g, new_b))
                else:
                    colored_image.putpixel((x, y), rgb_image.getpixel((x, y)))
        colored_image.save(saving_path)
    except Exception as e:
        print(f"Error processing {rgb_image_path} or {mask_image_path}: {e}")

def worker(args):
    rgb_image_path, mask_image_path, saving_path = args
    colorize_rgb_with_mask(rgb_image_path, mask_image_path, saving_path)

def main():
    args = parse_args()
    for video in args.video_list:
        for device in args.device_list:
            saving_path = f"{args.base_path}/{video}/device{device}/masked_image/"
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)
            tasks = []
            for filename in range(0, 2500):
                filename_str = f"{filename:06d}"
                rgb_image_path = f"{args.base_path}/{video}/device{device}/rect_left_camera/{filename_str}.jpg"
                mask_image_path = f"{args.base_path}/{video}/device{device}/mask/{filename_str}.png"
                save_path = os.path.join(saving_path, f'{filename_str}.jpg')
                tasks.append((rgb_image_path, mask_image_path, save_path))
            with Pool() as pool:
                list(tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc=f"{video}-device{device}", unit="img"))

if __name__ == '__main__':
    main()

