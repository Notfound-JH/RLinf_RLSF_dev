#!/usr/bin/env python3
"""
Extract episode indices from parquet filenames and save them to a CSV file.

This script reads parquet files from the source directory, extracts episode indices
from filenames (episode_XXXXX.parquet), and saves them to CSV, ensuring at most k
files are kept per task_index.
"""
import argparse
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import pandas as pd


def extract_episode_index(parquet_path: Path) -> int:
    """
    Extract the episode index from a parquet filename.

    Args:
        parquet_path: Path to the parquet file

    Returns:
        The episode index extracted from the filename, or -1 if not found
    """
    # Match pattern: episode_{digits}.parquet
    match = re.search(r'episode_(\d+)\.parquet$', parquet_path.name)
    if match:
        return int(match.group(1))
    return -1


def read_task_index(parquet_path: str) -> int:
    """
    Read the task_index from a parquet file.

    Args:
        parquet_path: Path to the parquet file

    Returns:
        The task_index value (assumed to be constant across the file)
    """
    try:
        df = pd.read_parquet(parquet_path)
        if 'task_index' not in df.columns:
            print(f"Warning: 'task_index' column not found in {parquet_path}, skipping...")
            return -1

        # Get the first task_index value (should be constant across the file)
        task_index = df['task_index'].iloc[0]
        return int(task_index)
    except Exception as e:
        print(f"Error reading {parquet_path}: {e}")
        return -1


def extract_and_save_indices(
    source_dir: str,
    output_csv: str,
    k: int = 1
) -> dict:
    """
    Extract episode indices from parquet files, limiting to k files per task_index,
    and save to CSV.

    Args:
        source_dir: Source directory containing parquet files
        output_csv: Path to output CSV file
        k: Maximum number of files to keep per task_index (default: 1)

    Returns:
        Dictionary with statistics about the operation
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Statistics
    stats = {
        'total_files': 0,
        'saved_files': 0,
        'skipped_files': 0,
        'error_files': 0,
        'tasks_found': 0,
        'task_counts': defaultdict(int)
    }

    # Find all parquet files recursively
    parquet_files = list(source_path.rglob('*.parquet'))
    stats['total_files'] = len(parquet_files)

    print(f"Found {stats['total_files']} parquet files in {source_dir}")
    print(f"Keeping at most {k} file(s) per task_index")
    print(f"Output CSV: {output_csv}")
    print("-" * 60)

    # Track all unique task_indices we've seen
    all_task_indices = set()
    early_stop_threshold_reached = False

    # Store results: list of (task_index, episode_index) tuples
    results = []

    for parquet_file in tqdm(parquet_files):
        # Check if all task_indices have reached k files
        if (all_task_indices and
            all(stats['task_counts'][task_idx] >= k for task_idx in all_task_indices) and
            len(all_task_indices) == 40):
            early_stop_threshold_reached = True
            break

        # Read task_index
        task_index = read_task_index(str(parquet_file))

        if task_index == -1:
            stats['error_files'] += 1
            continue

        # Extract episode index from filename
        episode_index = extract_episode_index(parquet_file)
        if episode_index == -1:
            stats['error_files'] += 1
            continue

        # Track this task_index
        all_task_indices.add(task_index)

        # Check if we've already kept k files for this task_index
        if stats['task_counts'][task_index] >= k:
            stats['skipped_files'] += 1
            continue

        # Store the result
        rel_path = parquet_file.relative_to(source_path)
        results.append({
            'task_index': task_index,
            'episode_index': episode_index,
            'filename': parquet_file.name
        })

        print(f"Keeping: {rel_path} (task_index={task_index}, episode_index={episode_index}, file {stats['task_counts'][task_index] + 1}/{k} for this task)")

        stats['saved_files'] += 1
        stats['task_counts'][task_index] += 1

    stats['tasks_found'] = len(stats['task_counts'])

    # Sort results by episode_index
    results.sort(key=lambda x: x['episode_index'])

    # Save episode indices to CSV
    episode_indices = np.array([res['episode_index'] for res in results], dtype=np.int32)
    np.savetxt(output_csv, episode_indices[None, :], delimiter=',', fmt='%d')

    print("-" * 60)
    if early_stop_threshold_reached:
        print(f"Early stop: All task_indices have reached {k} files")
        print(f"Stopped after processing {stats['saved_files'] + stats['skipped_files'] + stats['error_files']}/{stats['total_files']} files")
    print(f"Summary:")
    print(f"  Total files found: {stats['total_files']}")
    print(f"  Unique task_index values: {stats['tasks_found']}")
    print(f"  Files saved to CSV: {stats['saved_files']}")
    print(f"  Files skipped: {stats['skipped_files']}")
    print(f"  Files with errors: {stats['error_files']}")
    print(f"\nFiles per task_index:")
    for task_idx in sorted(stats['task_counts'].keys()):
        print(f"  task_index={task_idx}: {stats['task_counts'][task_idx]} file(s)")
    print(f"\nSaved {len(results)} episode indices to {output_csv}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract episode indices from parquet filenames based on task_index limit"
    )
    parser.add_argument(
        '--source',
        type=str,
        default='/workspace/public_dataset/physical-intelligence/libero/data/',
        help='Source directory containing parquet files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/workspace/public_dataset/physical-intelligence/libero/five_shot.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '-k',
        '--files-per-task',
        type=int,
        default=5,
        help='Maximum number of files to keep per task_index (default: 2)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Episode Index Extraction Script")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Output CSV: {args.output}")
    print(f"Files per task_index: {args.files_per_task}")
    print("=" * 60)
    print()

    try:
        extract_and_save_indices(
            source_dir=args.source,
            output_csv=args.output,
            k=args.files_per_task
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
