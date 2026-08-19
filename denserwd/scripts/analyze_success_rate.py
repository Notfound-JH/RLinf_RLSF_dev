#!/usr/bin/env python3
"""
Analyze success rate of trajectories in Libero HDF5 datasets.
This script reads all hdf5 files in data/datasets/libero_*/ directories,
extracts the final reward from each trajectory, and calculates the success rate.
"""

import os
import glob
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict


def analyze_hdf5_file(hdf5_path):
    """
    Analyze a single hdf5 file and extract final rewards from all trajectories.

    Args:
        hdf5_path: Path to the hdf5 file

    Returns:
        A dictionary with:
            - total_trajectories: total number of trajectories in the file
            - successful_trajectories: number of trajectories with final reward == 1
            - success_rate: success rate ratio
    """
    results = {
        'total_trajectories': 0,
        'successful_trajectories': 0,
        'failed_trajectories': 0,
        'success_rate': 0.0,
    }

    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Get all trajectory keys
            traj_keys = sorted([k for k in f.keys() if k.startswith('traj_')])
            results['total_trajectories'] = len(traj_keys)

            for traj_key in traj_keys:
                rewards = f[traj_key]['rewards'][:]
                final_reward = rewards[-1]
                # print(traj_key, final_reward)

                if final_reward == 1.0:
                    results['successful_trajectories'] += 1
                else:
                    results['failed_trajectories'] += 1
            print()
            if results['total_trajectories'] > 0:
                results['success_rate'] = results['successful_trajectories'] / results['total_trajectories']

    except Exception as e:
        print(f"Error reading {hdf5_path}: {e}")
        return None

    return results


def modify_reward_hdf5_file(hdf5_path):
    try:
        with h5py.File(hdf5_path, 'r+') as f:
            # Get all trajectory keys
            traj_keys = sorted([k for k in f.keys() if k.startswith('traj_')])

            for traj_key in traj_keys:
                if f[traj_key]["rewards"][-1] >= 1.0:
                    f[traj_key]["rewards"][-1] = 1.0
    except Exception as e:
        print(f"Error reading {hdf5_path}: {e}")
        return None


def main():
    # Find all libero dataset directories
    base_dir = Path('data/datasets')
    # libero_dirs = sorted(base_dir.glob('libero_*'))
    libero_dirs = [base_dir / "libero_goal_collect_openpi"]

    if not libero_dirs:
        print(f"No libero_* directories found in {base_dir}")
        return

    print("=" * 80)
    print("Libero Dataset Success Rate Analysis")
    print("=" * 80)

    # Aggregate statistics
    all_results = defaultdict(lambda: {
        'files': [],
        'total_trajectories': 0,
        'successful_trajectories': 0,
    })

    # Process each directory
    for libero_dir in libero_dirs:
        print(f"\n--- Processing: {libero_dir.name} ---")

        # Find all hdf5 files in this directory
        hdf5_files = sorted(libero_dir.glob('*.hdf5'))

        if not hdf5_files:
            print(f"  No hdf5 files found in {libero_dir}")
            continue

        print(f"  Found {len(hdf5_files)} hdf5 file(s)")

        # Process each file
        for hdf5_file in hdf5_files:
            # modify_reward_hdf5_file(hdf5_file)
            result = analyze_hdf5_file(hdf5_file)
            if result is None:
                continue

            # Store result
            all_results[libero_dir.name]['files'].append({
                'filename': hdf5_file.name,
                **result
            })
            all_results[libero_dir.name]['total_trajectories'] += result['total_trajectories']
            all_results[libero_dir.name]['successful_trajectories'] += result['successful_trajectories']

            # Print per-file result
            success_pct = result['success_rate'] * 100
            print(f"  [{hdf5_file.name}]")
            print("     Task: ", hdf5_file.stem.replace('_', ' '))
            print(f"    Trajectories: {result['total_trajectories']}, "
                  f"Success: {result['successful_trajectories']}, "
                  f"Failed: {result['failed_trajectories']}, "
                  f"Success Rate: {success_pct:.1f}%")

        # Print directory summary
        dir_total = all_results[libero_dir.name]['total_trajectories']
        dir_success = all_results[libero_dir.name]['successful_trajectories']
        dir_success_rate = (dir_success / dir_total * 100) if dir_total > 0 else 0
        print(f"  >>> {libero_dir.name} Summary: {dir_success}/{dir_total} "
              f"trajectories successful ({dir_success_rate:.1f}%)")

    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    grand_total = sum(r['total_trajectories'] for r in all_results.values())
    grand_success = sum(r['successful_trajectories'] for r in all_results.values())
    grand_success_rate = (grand_success / grand_total * 100) if grand_total > 0 else 0

    print(f"\nTotal datasets: {len(all_results)}")
    print(f"Total trajectories: {grand_total}")
    print(f"Total successful: {grand_success}")
    print(f"Overall success rate: {grand_success_rate:.2f}%")

    # Per-dataset summary table
    print("\n" + "-" * 80)
    print(f"{'Dataset':<40} {'Trajectories':>15} {'Successful':>12} {'Rate':>10}")
    print("-" * 80)

    for dataset_name in sorted(all_results.keys()):
        r = all_results[dataset_name]
        rate = (r['successful_trajectories'] / r['total_trajectories'] * 100) if r['total_trajectories'] > 0 else 0
        print(f"{dataset_name:<40} {r['total_trajectories']:>15} "
              f"{r['successful_trajectories']:>12} {rate:>9.1f}%")

    print("=" * 80)


if __name__ == '__main__':
    main()
