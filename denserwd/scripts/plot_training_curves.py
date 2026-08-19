#!/usr/bin/env python3
"""
Plot training curves from log files.
Extracts loss and MSE for each epoch and plots loss vs epoch and RMSE vs epoch.
"""

import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


SAVE_DIR = "./data/rwd_models/figures"


def parse_log_file(log_path):
    """
    Parse log file to extract epoch, loss, and MSE values.

    Args:
        log_path: Path to the log file

    Returns:
        tuple: (epochs, losses, mses) lists
    """
    epochs = []
    losses = []
    mses = []

    # Pattern to match epoch lines
    # Example: [Epoch   1] Loss: -0.0020 (MSE: 0.0003, Entropy: 0.3453)
    pattern = r'\[Epoch\s+(\d+)\]\s+Loss:\s+([-+]?\d+\.\d+)\s+\(MSE:\s+([-+]?\d+\.\d+)'

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                mse = float(match.group(3))

                epochs.append(epoch)
                losses.append(loss)
                mses.append(mse)

    return epochs, losses, mses


def plot_training_curves(epochs, losses, mses, output_path=None):
    """
    Plot loss vs epoch and RMSE vs epoch curves.

    Args:
        epochs: List of epoch numbers
        losses: List of loss values
        mses: List of MSE values
        output_path: Optional path to save the figure
    """
    # Convert MSE to RMSE
    rmses = np.sqrt(mses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss vs Epoch
    ax1.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot RMSE vs Epoch
    ax2.plot(epochs, rmses, 'r-', linewidth=2, marker='s', markersize=4, label='RMSE')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('RMSE vs Epoch', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves from log files')
    parser.add_argument('--log_file', type=str, help='Path to the log file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path to save the figure (optional)')
    parser.add_argument('--show-stats', action='store_true',
                        help='Show statistics about the training data')

    args = parser.parse_args()

    # Parse log file
    args.log_file = "/workspace/RLinf/denserwd/libero_40_cl.log"
    print(f"Parsing log file: {args.log_file}")
    epochs, losses, mses = parse_log_file(args.log_file)

    if not epochs:
        print("Error: No epoch data found in log file!")
        return

    print(f"Found {len(epochs)} epochs of training data")

    if args.show_stats:
        rmses = np.sqrt(mses)
        print(f"\nTraining Statistics:")
        print(f"  Epochs: {min(epochs)} to {max(epochs)}")
        print(f"  Loss range: [{min(losses):.6f}, {max(losses):.6f}]")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  MSE range: [{min(mses):.6f}, {max(mses):.6f}]")
        print(f"  RMSE range: [{min(rmses):.6f}, {max(rmses):.6f}]")
        print(f"  Final RMSE: {rmses[-1]:.6f}")

    if args.output is None:
        args.output = Path(SAVE_DIR) / (Path(args.log_file).stem + '_training_curves.png')

    # Plot curves
    plot_training_curves(epochs, losses, mses, args.output)


if __name__ == '__main__':
    main()
