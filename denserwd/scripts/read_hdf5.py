#!/usr/bin/env python3
"""
Utility script to read and inspect HDF5 dataset files.
"""

import sys
import os

import h5py
import argparse
import numpy as np
import glob
from libero.libero import get_libero_path


def print_structure(name, obj, indent=0):
    """Recursively print HDF5 structure."""
    prefix = "  " * indent
    if isinstance(obj, h5py.Dataset):
        print(f"{prefix}{name}: Dataset (shape={obj.shape}, dtype={obj.dtype})")
    elif isinstance(obj, h5py.Group):
        print(f"{prefix}{name}: Group")
        for key in obj.keys():
            print_structure(key, obj[key], indent + 1)


def analyze_all_hdf5_files(directory_path, detailed=False, demo_index=None):
    """
    Analyze all HDF5 files in a directory and find min/max episode lengths with percentiles.

    Args:
        directory_path: Path to the directory containing HDF5 files
        detailed: If True, print detailed information about each dataset
        demo_index: If specified, only show details for this demo index
    """
    # Find all HDF5 files in the directory
    hdf5_files = glob.glob(os.path.join(directory_path, "*.hdf5"))

    if not hdf5_files:
        print(f"Error: No HDF5 files found in {directory_path}")
        return

    print(f"{'='*80}")
    print(f"Found {len(hdf5_files)} HDF5 files in {directory_path}")
    print(f"{'='*80}")
    print()

    # Collect all episode lengths for percentile calculations
    all_episode_lengths = []
    episode_info_list = []  # Keep track of (file, demo, length) for all episodes

    # Global statistics across all files
    global_min_len = float('inf')
    global_max_len = 0
    global_min_info = {"file": None, "demo": None, "length": None}
    global_max_info = {"file": None, "demo": None, "length": None}

    # Process each file
    for file_idx, file_path in enumerate(sorted(hdf5_files), 1):
        print(f"[{file_idx}/{len(hdf5_files)}] Processing: {os.path.basename(file_path)}")

        file_min_len = float('inf')
        file_max_len = 0
        file_min_info = {"demo": None, "length": None}
        file_max_info = {"demo": None, "length": None}
        file_episode_lengths = []

        try:
            with h5py.File(file_path, 'r') as f:
                if 'data' not in f:
                    print(f"  Warning: No 'data' group found in file.")
                    continue

                data_group = f['data']
                demo_keys = sorted(data_group.keys(), key=lambda x: int(x.split('_')[1]))

                # Analyze each demo
                for demo_name in demo_keys:
                    if demo_index is not None:
                        idx = int(demo_name.split('_')[1])
                        if idx != demo_index:
                            continue

                    demo = data_group[demo_name]

                    # Get episode length from actions (usually the main dataset)
                    if 'actions' in demo:
                        actions = demo['actions']
                        episode_len = actions.shape[0]

                        # Add to collections
                        all_episode_lengths.append(episode_len)
                        episode_info_list.append({
                            "file": os.path.basename(file_path),
                            "demo": demo_name,
                            "length": episode_len
                        })
                        file_episode_lengths.append(episode_len)

                        # Update file-level statistics
                        if episode_len < file_min_len:
                            file_min_len = episode_len
                            file_min_info = {"demo": demo_name, "length": episode_len}

                        if episode_len > file_max_len:
                            file_max_len = episode_len
                            file_max_info = {"demo": demo_name, "length": episode_len}

                        # Update global statistics
                        if episode_len < global_min_len:
                            global_min_len = episode_len
                            global_min_info = {
                                "file": os.path.basename(file_path),
                                "demo": demo_name,
                                "length": episode_len
                            }

                        if episode_len > global_max_len:
                            global_max_len = episode_len
                            global_max_info = {
                                "file": os.path.basename(file_path),
                                "demo": demo_name,
                                "length": episode_len
                            }

                    # Print detailed info if requested
                    if detailed:
                        print(f"  {demo_name}: length = {episode_len}")

        except Exception as e:
            print(f"  Error processing file: {e}")
            continue

        # Print file-level summary with percentiles
        print(f"  File summary:")
        print(f"    Min episode: {file_min_info['demo']} (length={file_min_info['length']})")
        print(f"    Max episode: {file_max_info['demo']} (length={file_max_info['length']})")

        if file_episode_lengths:
            file_lengths_array = np.array(file_episode_lengths)
            print(f"    Percentiles:")
            print(f"      90%: {np.percentile(file_lengths_array, 90):.1f}")
            print(f"      95%: {np.percentile(file_lengths_array, 95):.1f}")
            print(f"      98%: {np.percentile(file_lengths_array, 98):.1f}")
            print(f"      99%: {np.percentile(file_lengths_array, 99):.1f}")
        print()

    # Calculate global percentiles
    if all_episode_lengths:
        all_lengths_array = np.array(all_episode_lengths)
        p90 = np.percentile(all_lengths_array, 90)
        p95 = np.percentile(all_lengths_array, 95)
        p98 = np.percentile(all_lengths_array, 98)
        p99 = np.percentile(all_lengths_array, 99)

        # Find episodes closest to each percentile
        def find_closest_to_percentile(target_length):
            closest = min(episode_info_list, key=lambda x: abs(x['length'] - target_length))
            return closest

        p90_info = find_closest_to_percentile(p90)
        p95_info = find_closest_to_percentile(p95)
        p98_info = find_closest_to_percentile(p98)
        p99_info = find_closest_to_percentile(p99)
    else:
        p90 = p95 = p98 = p99 = None
        p90_info = p95_info = p98_info = p99_info = None

    # Print global summary
    print(f"{'='*80}")
    print(f"GLOBAL SUMMARY across all {len(hdf5_files)} files")
    print(f"Total episodes: {len(all_episode_lengths)}")
    print(f"{'='*80}")
    print(f"Minimum episode:")
    print(f"  File: {global_min_info['file']}")
    print(f"  Demo: {global_min_info['demo']}")
    print(f"  Length: {global_min_info['length']}")
    print()
    print(f"Maximum episode:")
    print(f"  File: {global_max_info['file']}")
    print(f"  Demo: {global_max_info['demo']}")
    print(f"  Length: {global_max_info['length']}")
    print()
    print(f"Percentiles:")
    print(f"  90%: {p90:.1f}")
    print(f"    Closest episode: {p90_info['demo']} in {p90_info['file']} (length={p90_info['length']})")
    print()
    print(f"  95%: {p95:.1f}")
    print(f"    Closest episode: {p95_info['demo']} in {p95_info['file']} (length={p95_info['length']})")
    print()
    print(f"  98%: {p98:.1f}")
    print(f"    Closest episode: {p98_info['demo']} in {p98_info['file']} (length={p98_info['length']})")
    print()
    print(f"  99%: {p99:.1f}")
    print(f"    Closest episode: {p99_info['demo']} in {p99_info['file']} (length={p99_info['length']})")


def inspect_hdf5(file_path, detailed=False, demo_index=None):
    """
    Inspect an HDF5 file containing demonstration data.

    Args:
        file_path: Path to the HDF5 file
        detailed: If True, print detailed information about each dataset
        demo_index: If specified, only show details for this demo index
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"{'='*60}")
    print(f"HDF5 File: {file_path}")
    print(f"{'='*60}")
    print()

    min_episode_len = float('inf')
    max_episode_len = 0

    with h5py.File(file_path, 'r') as f:
        # Print top-level structure
        print("=== File Structure ===")
        print_structure('root', f)
        print()

        if 'data' not in f:
            print("No 'data' group found in file.")
            return

        data_group = f['data']
        demo_keys = sorted(data_group.keys(), key=lambda x: int(x.split('_')[1]))

        print(f"=== Dataset Overview ===")
        print(f"Number of demonstrations: {len(demo_keys)}")
        print(f"Demonstration names: {demo_keys[:5]}..." if len(demo_keys) > 5 else f"Demonstration names: {demo_keys}")
        print()

        # Analyze each demo
        for demo_name in demo_keys:
            if demo_index is not None:
                idx = int(demo_name.split('_')[1])
                if idx != demo_index:
                    continue

            print(f"--- {demo_name} ---")
            demo = data_group[demo_name]

            # Print available keys
            print(f"Keys: {list(demo.keys())}")
            print()

            # Get episode length
            if 'actions' in demo:
                episode_len = demo['actions'].shape[0]
                min_episode_len = min(min_episode_len, episode_len)
                max_episode_len = max(max_episode_len, episode_len)
                print(f"  Episode length: {episode_len}")
                print()

            # Print details for each key
            for key in sorted(demo.keys()):
                if key != 'rewards':
                    continue
                obj = demo[key]

                if isinstance(obj, h5py.Dataset):
                    data = obj[:]

                    print(f"  {key}:")
                    print(f"    Shape: {data.shape}")
                    print(f"    Dtype: {data.dtype}")
                    print(f"    Min-Max: {data.min():.4f} to {data.max():.4f}")

                    if detailed:
                        if data.ndim == 1 and data.size <= 20:
                            print(f"    Values: {data}")
                        elif data.ndim == 1:
                            print(f"    Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}")
                            print(f"    First 10: {data[:10]}")
                            print(f"    Last 10: {data[-10:]}")
                        elif data.ndim == 2 and data.shape[0] <= 5:
                            print(f"    Data preview:")
                            for i, row in enumerate(data):
                                print(f"      [{i}]: {row[:5]}..." if len(row) > 5 else f"      [{i}]: {row}")
                        else:
                            print(f"    Data preview (first element along each dim):")
                            print(f"      [0]: {data[0, ...][:5] if data.ndim > 1 else data[0]}")

                        # Special handling for rewards
                        if key == 'rewards':
                            non_zero = (data != 0).sum()
                            print(f"    Non-zero rewards: {non_zero}")
                            print(f"    Reward sum: {data.sum()}")

                            # Find indices with non-zero rewards
                            if non_zero > 0:
                                nonzero_indices = np.where(data != 0)[0]
                                print(f"    Non-zero indices: {nonzero_indices}")
                    print()

                elif isinstance(obj, h5py.Group):
                    print(f"  {key}: Group")
                    if detailed:
                        for subkey in obj.keys():
                            subobj = obj[subkey]
                            if isinstance(subobj, h5py.Dataset):
                                print(f"    {subkey}: shape={subobj.shape}, dtype={subobj.dtype}")
                    print()

            print()
            if demo_index is not None:
                break

    print(f"=== Summary ===")
    print(f"Minimum episode length: {min_episode_len}")
    print(f"Maximum episode length: {max_episode_len}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect HDF5 demonstration files or analyze all files in a directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all HDF5 files in a directory
  python read_hdf5.py --dir /path/to/directory

  # Analyze with detailed output
  python read_hdf5.py --dir /path/to/directory --detailed

  # Inspect a single file
  python read_hdf5.py --file /path/to/file.hdf5

  # Use default LIBERO dataset
  python read_hdf5.py --benchmark libero_spatial
        """
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Path to a single HDF5 file (relative or absolute). Use with --file flag to inspect one file.'
    )
    parser.add_argument(
        '--dir',
        '--directory',
        type=str,
        dest='directory',
        help='Path to directory containing HDF5 files. Analyzes all files in the directory.'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        default='libero_goal',
        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'],
        help='Benchmark name (used if --file and --dir not specified). Default: libero_spatial'
    )
    parser.add_argument(
        '--demo',
        type=int,
        default=None,
        help='Only show details for specific demo index'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed information about each dataset'
    )

    args = parser.parse_args()
    args.file = 'denserwd/data/libero_spatial_collect_openpi_data.hdf5'
    # Determine what to analyze
    if args.file:
        # Single file mode
        file_path = args.file if os.path.isabs(args.file) else os.path.join(os.getcwd(), args.file)
        inspect_hdf5(file_path, detailed=args.detailed, demo_index=args.demo)
    elif args.directory:
        # Directory mode - analyze all files
        analyze_all_hdf5_files(args.directory, detailed=args.detailed, demo_index=args.demo)
    else:
        # Default mode - use LIBERO benchmark dataset
        dataset_folder = get_libero_path('datasets')
        benchmark_dir = os.path.join(dataset_folder, args.benchmark)
        if os.path.exists(benchmark_dir):
            print(f"Analyzing all HDF5 files in: {benchmark_dir}\n")
            analyze_all_hdf5_files(benchmark_dir, detailed=args.detailed, demo_index=args.demo)
        else:
            print(f"Error: Directory not found: {benchmark_dir}")


if __name__ == '__main__':
    main()
