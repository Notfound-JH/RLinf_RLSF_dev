import random
import torch
import os
import numpy as np


def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_hdf5_files(directory, recursive=False):
    """
    Get all HDF5 files in the target directory.

    Args:
        directory (str): Path to the directory to search
        recursive (bool): If True, search recursively in subdirectories. Default is True.

    Returns:
        list: Sorted list of paths to all HDF5 files found

    Examples:
        >>> files = get_hdf5_files("/path/to/data")
        >>> files = get_hdf5_files("/path/to/data", recursive=False)
    """
    hdf5_files = []

    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.hdf5') or file.endswith('.h5'):
                    hdf5_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and (file.endswith('.hdf5') or file.endswith('.h5')):
                hdf5_files.append(file_path)

    return sorted(hdf5_files)
