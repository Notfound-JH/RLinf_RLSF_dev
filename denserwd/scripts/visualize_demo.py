#!/usr/bin/env python3
"""
Visualize demonstration trajectories from HDF5 dataset with synchronized reward plots.

This script creates a video with two panels:
- Left: Trajectory visualization (image observations)
- Right: Reward curve over time (synchronized with video frames)

Usage:
    python scripts/visualize_demo.py --hdf5_path <path> --demo_id <demo_0> --output <output.mp4>
"""

import argparse
import os
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
from tqdm import tqdm
import imageio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from denserwd.datasets.datasets import SequenceDataset
from denserwd.utils.data_utils import load_demo_data


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize demo trajectory with rewards")
    parser.add_argument("--hdf5_path", type=str, default="/home/wuduo/projects/LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5",
                        help="Path to HDF5 dataset file")
    parser.add_argument("--demo_id", type=str, default="demo_0",
                        help="Demo ID to visualize (e.g., demo_0, demo_1)")
    parser.add_argument("--output", type=str, default="demo_visualization.mp4",
                        help="Output video path")
    parser.add_argument("--fps", type=int, default=20,
                        help="Frames per second for output video")
    parser.add_argument("--image_key", type=str, default="agentview_rgb",
                        help="Key for image observations in HDF5")
    parser.add_argument("--obs_modality", type=str, default="null",
                        help="Observation modality (for ObsUtils initialization)")
    parser.add_argument("--reward_key", type=str, default="rewards",
                        help="Key for rewards in HDF5")
    parser.add_argument("--height", type=int, default=480,
                        help="Height of output video")
    parser.add_argument("--width", type=int, default=1280,
                        help="Width of output video")
    parser.add_argument("--show_actions", action="store_true",
                        help="Show action values on the plot")
    return parser.parse_args()


def create_visualization_video(demo_data, output_path, fps=15, height=480, width=1280, show_actions=False, predicted_rewards=None, flip_image=False):
    """
    Create visualization video with trajectory and reward plot using imageio.

    Args:
        demo_data: Dictionary containing demo data
        output_path: Path to save output video
        fps: Frames per second
        height: Video height
        width: Video width
        show_actions: Whether to show action values
        predicted_rewards: Optional array of predicted rewards (T,) for comparison with ground truth
    """
    images = demo_data['images']
    rewards = demo_data['rewards']
    actions = demo_data['actions']
    task_description = demo_data.get('task_description', 'Task')

    num_samples = min(demo_data['num_samples'], predicted_rewards.shape[0]) if predicted_rewards is not None else num_samples
    images = images[:num_samples]
    rewards = rewards[:num_samples]
    actions = actions[:num_samples]

    # Validate predicted_rewards if provided
    if predicted_rewards is not None:
        predicted_rewards = np.array(predicted_rewards)
        predicted_rewards = predicted_rewards[:num_samples]
        print(f"Using predicted rewards for comparison")

    print(f"Creating visualization video with {num_samples} frames...")
    print(f"Task: {task_description}")

    # Calculate aspect ratio for image panel
    img_height, img_width = images.shape[1:3]
    aspect_ratio = img_width / img_height

    # Calculate figure size to match desired video dimensions
    # Total width = image_width + plot_width (approximately 30% of image width)
    total_width_inches = width / 100  # Convert pixels to inches at 100 dpi
    total_height_inches = height / 100

    # Initialize imageio writer
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')

    print("Generating frames...")
    for frame_idx in tqdm(range(num_samples)):
        # Create figure for this frame
        fig = plt.figure(figsize=(total_width_inches, total_height_inches), dpi=100)

        # Add overall title for task description at the top
        fig.suptitle(task_description, fontsize=14, fontweight='bold', y=0.98)

        # Create subplot grid with space for title
        gs = gridspec.GridSpec(1, 2, width_ratios=[aspect_ratio, 1], top=0.90, bottom=0.08,
                               left=0.03, right=0.98, wspace=0.15)

        # Image subplot (left)
        ax_img = fig.add_subplot(gs[0])
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        if flip_image:
            ax_img.imshow(images[frame_idx][::-1, :, :])  # Flip vertically for correct orientation
        else:
            ax_img.imshow(images[frame_idx])  

        # Reward subplot (right)
        ax_reward = fig.add_subplot(gs[1])
        ax_reward.set_xlim(0, max(num_samples, 10))

        # Adjust y-axis limits if we have predicted rewards
        if predicted_rewards is not None:
            predicted_rewards[-1] = rewards[-1]
            all_rewards = np.concatenate([rewards, predicted_rewards])
            y_min = max(0, all_rewards.min() - 0.1)
            y_max = min(1.1, all_rewards.max() + 0.1)
            ax_reward.set_ylim(y_min, y_max)
        else:
            ax_reward.set_ylim(0, 1.1)

        ax_reward.set_xlabel('Timestep', fontsize=10)
        ax_reward.set_ylabel('Reward', fontsize=10)

        ax_reward.set_title('Reward over Time', fontsize=12)

        ax_reward.grid(True, alpha=0.3)

        # Plot reward curves up to current frame
        timesteps = np.arange(num_samples)

        if predicted_rewards is None:
            # Only show ground truth rewards
            ax_reward.plot(timesteps[:frame_idx+1], rewards[:frame_idx+1],
                          'b-', linewidth=2, label='Original Reward')
            ax_reward.plot(frame_idx, rewards[frame_idx], 'bo', markersize=8, label='Current')

            # Add frame info
            frame_text = f'Frame: {frame_idx}/{num_samples-1}\nReward: {rewards[frame_idx]:.3f}'
        else:
            # Show both ground truth and predicted rewards
            ax_reward.plot(timesteps[:frame_idx+1], rewards[:frame_idx+1],
                          'g-', linewidth=2, label='Original Reward', alpha=0.7)
            ax_reward.plot(timesteps[:frame_idx+1], predicted_rewards[:frame_idx+1],
                          'b-', linewidth=2, label='Densifed Reward', alpha=0.7)

            # Current timestep markers
            ax_reward.plot(frame_idx, rewards[frame_idx], 'go', markersize=8, label='Current (Original)')
            ax_reward.plot(frame_idx, predicted_rewards[frame_idx], 'bo', markersize=8, label='Current (Densifed)')

            # Calculate error for current frame
            error = abs(predicted_rewards[frame_idx] - rewards[frame_idx])

            # Add frame info with comparison
            frame_text = (f'Frame: {frame_idx}/{num_samples-1}\n'
                         f'GT: {rewards[frame_idx]:.3f}\n'
                         f'Pred: {predicted_rewards[frame_idx]:.3f}\n'
                         f'Error: {error:.3f}')

        # Add legend
        ax_reward.legend(loc='upper left', fontsize=8, ncol=2)

        # Add action text if available
        if show_actions and actions is not None:
            action_str = f"Actions:\n[{', '.join([f'{x:.2f}' for x in actions[frame_idx][:4]])}, ...]"
            ax_reward.text(0.02, 0.98, action_str, transform=ax_reward.transAxes,
                          verticalalignment='top', fontsize=7,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Add frame info text box
        ax_img.text(0.02, 0.98, frame_text, transform=ax_img.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Convert figure to numpy array
        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # Reshape to image
        ncols, nrows = fig.canvas.get_width_height()
        frame_array = buf.reshape(nrows, ncols, 3)

        # Append to video
        writer.append_data(frame_array)

        plt.close(fig)

    # Close writer
    writer.close()
    print(f"Video saved successfully to {output_path}!")


def main():
    args = parse_args()

    # Check if HDF5 file exists
    if not os.path.exists(args.hdf5_path):
        print(f"Error: HDF5 file not found: {args.hdf5_path}")
        return

    # Load demo data
    demo_data = load_demo_data(
        args.hdf5_path,
        args.demo_id,
        args.image_key,
        args.reward_key
    )

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create visualization video
    try:
        create_visualization_video(
            demo_data,
            args.output,
            fps=args.fps,
            height=args.height,
            width=args.width,
            show_actions=args.show_actions
        )
        print("Done!")
    except Exception as e:
        print(f"Error creating video: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
