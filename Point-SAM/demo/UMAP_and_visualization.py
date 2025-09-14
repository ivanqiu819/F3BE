import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import umap
import torch
import argparse
import cv2

def natural_sort_key(s):
    """
    Sorts strings in a human-readable way, sorting numbers in the string numerically.
    """
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def visualize_single_group_scatter(embedding, base_path, embedding_name):
    fig = plt.figure(figsize=(16, 9), facecolor='white')

    ax_umap = fig.add_axes([0.07, 0.12, 0.68, 0.8], facecolor='white')
    ax_img = fig.add_axes([0.78, 0.18, 0.18, 0.32], facecolor='white')
    ax_slider = fig.add_axes([0.07, 0.05, 0.85, 0.06], facecolor='white')
    ax_img.axis('off')

    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(embedding.T)
        density = kde(embedding.T)
    except Exception:
        density = np.ones(embedding.shape[0])
    norm_density = (density - np.min(density)) / (np.max(density) - np.min(density) + 1e-8)
    cmap = plt.get_cmap('Blues')
    colors = cmap(norm_density)
    scatter = ax_umap.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=7,
        c=colors,
        alpha=0.7,
        picker=True
    )

    ax_umap.set_xticks([])
    ax_umap.set_yticks([])
    for spine in ax_umap.spines.values():
        spine.set_color('black')
    ax_umap.set_xlabel('UMAP Dimension 1', fontsize=25, color='black')
    ax_umap.set_ylabel('UMAP Dimension 2', fontsize=25, color='black')

    image_list = []
    image_dir = os.path.join(base_path, "device0/left_camera_rectified")
    image_names = sorted(os.listdir(image_dir), key=natural_sort_key)
    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        image_list.append(image_path)

    # Add title to the image
    ax_img.set_title('Original Image', fontsize=14, color='black')

    slider = widgets.Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=len(image_list)-1,
        valinit=0,
        valstep=1
    )

    current_idx = [0]  # use list for mutability in closures

    def update(idx):
        idx = int(idx)
        current_idx[0] = idx
        # Remove previous highlight if exists
        for coll in ax_umap.collections:
            if hasattr(coll, '_is_highlight') and coll._is_highlight:
                coll.remove()
        # Draw glowing highlight (simulate glow by stacking several black points)
        for i, (size, alpha) in enumerate(zip([500, 300, 150, 60], [0.15, 0.25, 0.4, 0.8])):
            glow = ax_umap.scatter(
                [embedding[idx, 0]], [embedding[idx, 1]],
                s=size, c="#222222", alpha=alpha, edgecolors='none', zorder=10+i
            )
            glow._is_highlight = True
        circle = ax_umap.scatter(
            [embedding[idx, 0]], [embedding[idx, 1]],
            s=700, facecolors='none', edgecolors='black', linewidths=2.5, zorder=20
        )
        circle._is_highlight = True
        # Show image
        ax_img.cla()
        ax_img.axis('off')
        ax_img.set_title('Original Image', fontsize=12, color='black')
        img = cv2.imread(image_list[idx])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            ax_img.imshow(img)
        fig.canvas.draw_idle()

    def on_slider(val):
        update(val)

    def on_key(event):
        idx = current_idx[0]
        if event.key == 'right' and idx < len(image_list) - 1:
            slider.set_val(idx + 1)
        elif event.key == 'left' and idx > 0:
            slider.set_val(idx - 1)

    slider.on_changed(on_slider)
    update(0)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def main(args):
    """
    Main function to run the full pipeline with dynamic color coding
    """
    embedding_dir = os.path.join(args.base_path, 'embeddings')
    embedding_name = f"norm={args.normalize}_center={args.centralize}_{args.rotate_num}feature_{args.pooling_type}"
    embedding_file = f"{embedding_name}.pth"
    frames = sorted(os.listdir(embedding_dir), key=natural_sort_key)
    print(f"Total frames: {len(frames)}")
    # load embeddings
    concat_features_list = []
    for frame in frames:
        frame_path = os.path.join(embedding_dir, frame)
        embedding_path = os.path.join(frame_path, embedding_file)
        if not os.path.exists(embedding_path):
            print(f"Embedding file {embedding_path} does not exist. Skipping frame {frame}.")
            continue
        pc_embeddings = torch.load(embedding_path, map_location='cpu')
        # print(f"Loaded embedding for frame {frame}: shape {pc_embeddings.shape}")
        concat_features_list.append(pc_embeddings)
    
    concat_features = torch.cat(concat_features_list, dim=1).squeeze(0).numpy()

    # UMAP dimensionality reduction
    umap_reducer = umap.UMAP(
        n_components=2,
        metric='cosine',
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    embedding = umap_reducer.fit_transform(concat_features)
    visualize_single_group_scatter(embedding, args.base_path, embedding_name)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding UMAP and Visualization")
    # Basic arguments
    parser.add_argument('--base_path', type=str,
                         default="/home/user/PycharmProjects/animal_pose_-gui/example_operator1/experiment/test",
                         help="Base path for point cloud and images.")
    parser.add_argument('--rotate_num', type=int, default=16, help="Number of rotations to apply on each point cloud.")
    parser.add_argument('--centralize', type=bool, default=True, help="Whether to centralize the point cloud.")
    parser.add_argument('--normalize', type=bool, default=True, help="Whether to normalize the point cloud.")
    parser.add_argument("--pooling_type", type=str, default="max", choices=["max", "mean"], help="Pooling type.")
    args = parser.parse_args()

    main(args)