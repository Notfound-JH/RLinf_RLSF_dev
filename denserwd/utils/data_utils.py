import h5py
import numpy as np
import os


def save_dict_to_hdf5(data, h5file, path=""):
    """
    递归将嵌套 dict 保存为 HDF5。
    - dict → group
    - np.ndarray → dataset (with compression)
    """
    if isinstance(data, dict):
        grp = h5file.create_group(path) if path else h5file
        for key, value in data.items():
            new_path = f"{path}/{key}" if path else key
            save_dict_to_hdf5(value, h5file, new_path)
    elif isinstance(data, np.ndarray):
        h5file.create_dataset(
            path,
            data=data,
            compression="gzip",       # 压缩节省空间
            compression_opts=9,       # 最高压缩率（可选 1-9）
            chunks=True               # 启用分块，利于部分读取
        )
    else:
        raise ValueError(f"Unsupported data type at {path}: {type(data)}")


def safe_save_traj_to_hdf5(h5file, traj_key, traj_data, overwrite=False):
    if traj_key in h5file:
        if overwrite:
            del h5file[traj_key]
        else:
            raise KeyError(f"Trajectory key '{traj_key}' already exists and overwrite=False")
    save_dict_to_hdf5(traj_data, h5file, traj_key)


def load_hdf5_to_dict(h5file, path=""):
    """从 HDF5 递归重建嵌套 dict"""
    obj = h5file[path] if path else h5file

    if isinstance(obj, h5py.Group):
        return {key: load_hdf5_to_dict(h5file, f"{path}/{key}") for key in obj.keys()}
    elif isinstance(obj, h5py.Dataset):
        return np.array(obj)  # 自动转为 np.ndarray
    else:
        raise RuntimeError(f"Unexpected object type: {type(obj)}")

def wrap_lerobot_batch_for_reward_model(batch):
    new_batch = {"obs": {}}
    obs_map = {"image": "main_images", "wrist_image": "wrist_images", "state": "robot_states"}
    for key, new_key in obs_map.items():
        new_batch["obs"][new_key] = batch[key]
    for key in batch.keys():
        if key not in obs_map:
            new_batch[key] = batch[key]
    return new_batch

def batch_image_hwc_to_chw(im):
    """
    Channel swap for images - useful for preparing images for
    torch training.

    Args:
        im (np.array or torch.Tensor): image of shape (batch, height, width, channel)
            or (height, width, channel)

    Returns:
        im (np.array or torch.Tensor): image of shape (batch, channel, height, width)
            or (channel, height, width)
    """
    start_dims = np.arange(len(im.shape) - 3).tolist()
    s = start_dims[-1] if len(start_dims) > 0 else -1
    if isinstance(im, np.ndarray):
        return im.transpose(start_dims + [s + 3, s + 1, s + 2])
    else:
        return im.permute(start_dims + [s + 3, s + 1, s + 2])


def load_demo_data(hdf5_path, demo_id, image_key="main_images", reward_key="rewards"):
    """
    Load demonstration data from HDF5 file.

    Supports both old format (data/demo_id/) and new flat format (traj_id/ at root).

    Args:
        hdf5_path: Path to HDF5 file
        demo_id: Demonstration ID (e.g., "demo_0", "traj_0")
        image_key: Key for image observations (e.g., "main_images", "agentview_rgb")
        reward_key: Key for rewards

    Returns:
        Dictionary containing demo data
    """
    print(f"Loading demo {demo_id} from {hdf5_path}...")

    # Extract task description from filename
    task_description = os.path.basename(hdf5_path).replace('.hdf5', '').replace('_', ' ').title()

    with h5py.File(hdf5_path, 'r') as f:
        # Try new format first (traj at root level)
        if demo_id in f:
            demo_group = f[demo_id]
            # New format uses obs_ prefix
            obs_image_key = f"obs_{image_key}"
            num_samples = demo_group["actions"].shape[0]
        # Fall back to old format (data/demo_id)
        elif f"data/{demo_id}" in f:
            demo_group = f[f"data/{demo_id}"]
            obs_image_key = f"obs/{image_key}"
            num_samples = demo_group.attrs["num_samples"]
        else:
            raise KeyError(f"Unable to find trajectory '{demo_id}' in HDF5 file. "
                          f"Tried both '{demo_id}' and 'data/{demo_id}'")

        print(f"  Number of samples: {num_samples}")

        # Load images
        images = demo_group[obs_image_key][:]
        print(f"  Image shape: {images.shape}")

        # Load rewards if available
        rewards = None
        if reward_key in demo_group:
            rewards = demo_group[reward_key][:]
            print(f"  Rewards shape: {rewards.shape}")
        else:
            print(f"  Warning: {reward_key} not found, using zeros")
            rewards = np.zeros(num_samples)

        # Load actions if available
        actions = None
        if "actions" in demo_group:
            actions = demo_group["actions"][:]
            print(f"  Actions shape: {actions.shape}")

        # Load dones if available
        dones = None
        if "dones" in demo_group:
            dones = demo_group["dones"][:]
            print(f"  Dones shape: {dones.shape}")

    return {
        'images': images,
        'rewards': rewards,
        'actions': actions,
        'dones': dones,
        'num_samples': num_samples,
        'task_description': task_description
    }

