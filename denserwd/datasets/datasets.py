# Override the default SequenceDataset in libero to to support new flat HDF5 format
import os
import h5py
import numpy as np
import torch.utils.data

from tqdm import tqdm
from contextlib import contextmanager
from denserwd.utils.data_utils import batch_image_hwc_to_chw
from denserwd.utils.tensor_utils import pad_sequence


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hdf5_path,
        obs_keys,
        dataset_keys,
        frame_stack=1,
        seq_length=1,
        require_target=True,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=None,
        hdf5_use_swmr=True,
        traj_keys=None,  
        load_next_obs=False,
        sample_step="auto",
        k=5,
    ):
        """
        Modified to support new flat HDF5 format:
          traj_0/
            obs_main_images, obs_wrist_images, obs_robot_states,
            next_obs_main_images, ..., actions, rewards, dones
        """
        super().__init__()

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.hdf5_use_swmr = hdf5_use_swmr
        self.k = k
        self._hdf5_file = None

        assert hdf5_cache_mode in ["all", "low_dim", None]
        self.hdf5_cache_mode = hdf5_cache_mode

        self.load_next_obs = load_next_obs
        self.traj_keys = traj_keys  

        # Validate obs_keys against expected new format
        expected_obs_keys = {"main_images", "wrist_images", "robot_states"}
        assert all(k in expected_obs_keys for k in obs_keys), f"obs_keys must be subset of {expected_obs_keys}"

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.require_target = require_target

        self.sample_step = sample_step
        if self.sample_step == "auto":
            self.sample_step = min(max(1, self.seq_length // 3), 15)
        assert self.sample_step >= 1

        self.goal_mode = goal_mode

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.load_traj_info(traj_list=self.traj_keys)

        # maybe store dataset in memory for fast access
        self.hdf5_cache_mode = None
        if self.hdf5_cache_mode in ["all", "low_dim"]:
            obs_keys_in_memory = self.obs_keys
            if self.hdf5_cache_mode == "low_dim":
                # In new format, only robot_states is low-dim
                obs_keys_in_memory = [k for k in self.obs_keys if k == "robot_states"]
            self.obs_keys_in_memory = obs_keys_in_memory

            self.hdf5_cache = self.load_dataset_in_memory(
                traj_list=self.trajs,
                hdf5_file=self.hdf5_file,
                obs_keys=self.obs_keys_in_memory,
                dataset_keys=self.dataset_keys,
                load_next_obs=self.load_next_obs,
            )

            if self.hdf5_cache_mode == "all":
                print("SequenceDataset: caching get_item calls...")
                self.getitem_cache = [
                    self.get_item(i) for i in tqdm(range(len(self)))
                ]
                del self.hdf5_cache
                self.hdf5_cache = None
        else:
            self.hdf5_cache = None

        self.close_and_delete_hdf5_handle()
        print("Sample step:", self.sample_step)
        print("Total number of sequences:", self.total_num_sequences)

    def load_traj_info(self, traj_list=None):
        """Load trajectory metadata from root level (not under /data)."""
        if traj_list is not None:
            self.trajs = list(sorted(traj_list, key=lambda x: int(x.split('_')[-1])))
        else:
            # List all traj_* groups at root
            all_keys = list(self.hdf5_file.keys())
            self.trajs = list(sorted([k for k in all_keys if k.startswith("traj_")], key=lambda x: int(x.split('_')[-1])))

        self.n_trajs = len(self.trajs)

        # keep internal index maps to know which transitions belong to which trajectory
        self._index_to_traj_id = {}  # maps every index to a trajectory id
        self._traj_id_to_start_indices = {}  # gives start index per trajectory id
        self._traj_id_to_traj_length = {}
        self._traj_id_to_final_reward = {}

        # determine index mapping
        self.total_num_sequences = 0
        for ep in self.trajs:
            traj_length = self.hdf5_file[ep]["actions"].shape[0] 
            self._traj_id_to_start_indices[ep] = self.total_num_sequences
            self._traj_id_to_traj_length[ep] = traj_length
            self._traj_id_to_final_reward[ep] = self.hdf5_file[ep]["rewards"][-1]

            num_sequences = traj_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_frame_stack:
                num_sequences -= (self.n_frame_stack - 1)
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            if self.pad_seq_length:
                assert traj_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                assert num_sequences >= 1

            # Ensure last sample is included
            end = num_sequences + self.sample_step
            for i in range(0, end, self.sample_step):
                self._index_to_traj_id[self.total_num_sequences] = ep
                self.total_num_sequences += 1

    @property
    def hdf5_file(self):
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest')
        return self._hdf5_file

    def close_and_delete_hdf5_handle(self):
        if self._hdf5_file is not None:
            self._hdf5_file.close()
            self._hdf5_file = None

    @contextmanager
    def hdf5_file_opened(self):
        should_close = self._hdf5_file is None
        yield self.hdf5_file
        if should_close:
            self.close_and_delete_hdf5_handle()

    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __len__(self):
        return self.total_num_sequences

    def load_dataset_in_memory(self, traj_list, hdf5_file, obs_keys, dataset_keys, load_next_obs):
        all_data = {}
        print("SequenceDataset: loading dataset into memory...")
        for ep in tqdm(traj_list):
            all_data[ep] = {}
            all_data[ep]["attrs"] = {}
            all_data[ep]["attrs"]["num_samples"] = hdf5_file[ep]["actions"].shape[0]

            # Load obs
            all_data[ep]["obs"] = {}
            for k in obs_keys:
                ds_name = f"obs_{k}"
                all_data[ep]["obs"][k] = hdf5_file[ep][ds_name][()].astype('float32')

            if load_next_obs:
                all_data[ep]["next_obs"] = {}
                for k in obs_keys:
                    ds_name = f"next_obs_{k}"
                    all_data[ep]["next_obs"][k] = hdf5_file[ep][ds_name][()].astype('float32')

            # Load other keys
            for k in dataset_keys:
                if k in hdf5_file[ep]:
                    all_data[ep][k] = hdf5_file[ep][k][()].astype('float32')
                else:
                    n = all_data[ep]["attrs"]["num_samples"]
                    all_data[ep][k] = np.zeros((n, 1), dtype=np.float32)

        return all_data


    def get_dataset_for_ep(self, ep, key):
        key_should_be_in_memory = (self.hdf5_cache_mode in ["all", "low_dim"])
        if key_should_be_in_memory:
            if '/' in key:
                prefix, obs_key = key.split('/', 1)
                if prefix == "obs":
                    if obs_key not in self.obs_keys_in_memory:
                        key_should_be_in_memory = False
                elif prefix == "next_obs":
                    if obs_key not in self.obs_keys_in_memory:
                        key_should_be_in_memory = False

        if key_should_be_in_memory:
            if '/' in key:
                prefix, obs_key = key.split('/', 1)
                return self.hdf5_cache[ep][prefix][obs_key]
            else:
                return self.hdf5_cache[ep][key]
        else:
            # Map logical key to physical dataset name
            if key.startswith("obs/"):
                _, obs_key = key.split("/", 1)
                hd5key = f"{ep}/obs_{obs_key}"
            elif key.startswith("next_obs/"):
                _, obs_key = key.split("/", 1)
                hd5key = f"{ep}/next_obs_{obs_key}"
            else:
                hd5key = f"{ep}/{key}"
            return self.hdf5_file[hd5key]

    def __getitem__(self, index):
        if self.hdf5_cache_mode == "all":
            return self.getitem_cache[index]
        return self.get_item(index)

    def get_item(self, index):
        traj_id = self._index_to_traj_id[index]
        traj_start_index = self._traj_id_to_start_indices[traj_id]
        traj_length = self._traj_id_to_traj_length[traj_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_traj = (index - traj_start_index + demo_index_offset) * self.sample_step
        index_in_traj = max(0, min(index_in_traj, traj_length - 1))
        target_index_in_traj = min(index_in_traj + 1, traj_length - 1)  # for target model

        # end at offset index if not padding for seq length
        traj_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_traj = traj_length - traj_length_offset

        meta = self.get_dataset_sequence_from_demo(
            traj_id, index_in_traj, self.dataset_keys, self.seq_length
        )
        if traj_length - index_in_traj <= self.k:
            meta["rewards"][-1] = self._traj_id_to_final_reward[traj_id]
            meta["dones"][-1] = True

        if self.require_target:
            target_meta = self.get_dataset_sequence_from_demo(
                traj_id, target_index_in_traj, self.dataset_keys, self.seq_length + 1
            )
            if traj_length - target_index_in_traj <= self.k:
                target_meta["rewards"][-1] = self._traj_id_to_final_reward[traj_id]
                target_meta["dones"][-1] = True

        goal_index = None
        if self.goal_mode == "last":
            goal_index = end_index_in_traj - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            traj_id, index_in_traj, self.obs_keys, self.n_frame_stack - 1, self.seq_length, prefix="obs"
        )

        if self.require_target:
            target_meta["obs"] = self.get_obs_sequence_from_demo(
                traj_id, target_index_in_traj, self.obs_keys, self.n_frame_stack - 1, self.seq_length + 1, prefix="obs"
            )

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                traj_id, index_in_traj, self.obs_keys, self.n_frame_stack - 1, self.seq_length, prefix="next_obs"
            )

        if goal_index is not None:
            goal = self.get_obs_sequence_from_demo(
                traj_id, goal_index, self.obs_keys, 0, 1, prefix="next_obs"
            )
            meta["goal_obs"] = {k: goal[k][0] for k in goal}

        # Add timesteps information
        # Timesteps represent the timestamp of each sample in the sequence (starting from 1, 0 for padding)
        meta["timesteps"] = self._generate_timesteps(index_in_traj, self.seq_length, traj_length)
        if self.require_target:
            target_meta["timesteps"] = self._generate_timesteps(target_index_in_traj, self.seq_length + 1, traj_length)
            meta["target_model"] = target_meta

        return meta

    def _generate_timesteps(self, index_in_traj, seq_len, traj_len):
        """
        Generate timesteps for a sequence starting at index_in_demo.

        Args:
            index_in_demo (int): the current index in the demo (end index of the sequence)
            seq_len (int): length of the sequence
            demo_length (int): total length of the demo

        Returns:
            np.ndarray: array of timesteps with shape (seq_length,), where each value represents
                       the timestep number (starting from 1), and 0 represents padding
        """
        # Sequence is [index_in_demo - seq_length + 1, index_in_demo + 1)
        # Current frame (index_in_demo) is the LAST frame in the sequence
        seq_begin_index = index_in_traj - seq_len + 1
        seq_end_index = index_in_traj + 1

        # Generate timesteps using vectorized operations
        timesteps = np.arange(seq_begin_index, seq_end_index, dtype=np.int64)

        # Set invalid indices to 0 (padding)
        # Valid range is [1, demo_length)
        timesteps[(timesteps < 1) | (timesteps >= traj_len)] = 0

        return timesteps

    def get_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        The sequence is extracted from [index_in_demo - seq_length + 1, index_in_demo + 1),
        i.e., the current frame is the LAST frame in the sequence, and we look back
        seq_length - 1 frames.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): end index of the sequence (current frame) wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets prepended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self._traj_id_to_traj_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        # Sequence is [index_in_demo - seq_length + 1, index_in_demo + 1)
        # Current frame (index_in_demo) is the LAST frame in the sequence
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack - seq_length + 1)
        seq_end_index = min(demo_length, index_in_demo - num_frames_to_stack + 1)

        # determine sequence padding (pad forward at the beginning)
        seq_begin_pad = max(0, num_frames_to_stack + seq_length - 1 - index_in_demo)  # pad at the beginning
        seq_end_pad = max(0, index_in_demo - num_frames_to_stack + 1 - demo_length)  # should be 0 since index_in_demo < demo_length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)
            seq[k] = data[seq_begin_index: seq_end_index].astype("float32")
        # Pad at the beginning (forward padding)
        seq = pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=False, pad_values=0.)
        if "dones" in seq:
            seq["dones"] = seq["dones"].astype(np.bool_)
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(np.bool_)

        return seq, pad_mask

    def get_obs_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1, prefix="obs"):
        full_keys = [f"{prefix}/{k}" for k in keys]
        obs, pad_mask = self.get_sequence_from_demo(
            demo_id, index_in_demo, full_keys, num_frames_to_stack, seq_length
        )
        obs = {k.split('/')[-1]: obs[k] for k in obs}
        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask
        for key in obs.keys():
            if "images" in key:
                obs[key] = batch_image_hwc_to_chw(obs[key])
        return obs

    def get_dataset_sequence_from_demo(self, demo_id, index_in_demo, keys, seq_length=1):
        data, pad_mask = self.get_sequence_from_demo(
            demo_id, index_in_demo, keys, 0, seq_length
        )
        if self.get_pad_mask:
            data["pad_mask"] = pad_mask
        return data

    def get_trajectory_at_index(self, index):
        traj_id = self.trajs[index]
        traj_length = self._traj_id_to_traj_length[traj_id]
        meta = self.get_dataset_sequence_from_demo(traj_id, 0, self.dataset_keys, traj_length)
        meta["obs"] = self.get_obs_sequence_from_demo(traj_id, 0, self.obs_keys, seq_length=traj_length)
        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(traj_id, 0, self.obs_keys, seq_length=traj_length, prefix="next_obs")
        meta["ep"] = traj_id
        return meta

    def get_dataset_sampler(self):
        return None


class BoundarySequenceDataset(SequenceDataset):
    def __init__(self, k=5, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self._rebuild_index_maps()

    def _rebuild_index_maps(self):
        self._index_to_traj_id = {}
        self._traj_id_to_start_indices = {}
        self._index_to_traj_index = {}
        self.total_num_sequences = 0

        for ep in self.trajs:
            traj_length = self._traj_id_to_traj_length[ep]
            if self.k >= traj_length:
                valid_indices = list(range(traj_length))
            else:
                last_k = list(range(traj_length - self.k, traj_length))
                valid_indices = last_k

            self._traj_id_to_start_indices[ep] = self.total_num_sequences
            for idx_in_traj in valid_indices:
                self._index_to_traj_id[self.total_num_sequences] = ep
                self._index_to_traj_index[self.total_num_sequences] = idx_in_traj
                self.total_num_sequences += 1

    def get_item(self, index):
        traj_id = self._index_to_traj_id[index]
        traj_length = self._traj_id_to_traj_length[traj_id]
        index_in_traj = self._index_to_traj_index[index]

        traj_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_traj = traj_length - traj_length_offset

        meta = self.get_dataset_sequence_from_demo(
            traj_id, index_in_traj, self.dataset_keys, self.seq_length
        )

        # # Set reward to final_reward for last k samples
        if traj_length - index_in_traj <= self.k:
            meta["rewards"][-1] = self._traj_id_to_final_reward[traj_id]
            meta["dones"][-1] = True

        goal_index = None
        if self.goal_mode == "last":
            goal_index = end_index_in_traj - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            traj_id, index_in_traj, self.obs_keys, self.n_frame_stack - 1, self.seq_length, prefix="obs"
        )

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                traj_id, index_in_traj, self.obs_keys, self.n_frame_stack - 1, self.seq_length, prefix="next_obs"
            )

        if goal_index is not None:
            goal = self.get_obs_sequence_from_demo(
                traj_id, goal_index, self.obs_keys, 0, 1, prefix="next_obs"
            )
            meta["goal_obs"] = {k: goal[k][0] for k in goal}

        meta["timesteps"] = self._generate_timesteps(index_in_traj, self.seq_length, traj_length)
        return meta

    def __repr__(self):
        msg = str(self.__class__.__name__)
        msg += " (\n\tpath={}\n\tobs_keys={}\n\tseq_length={}\n\tframe_stack={}\n"
        msg += "\tpad_seq_length={}\n\tpad_frame_stack={}\n\tgoal_mode={}\n"
        msg += "\tcache_mode={}\n\tk={}\n\tnum_trajs={}\n\tnum_sequences={}\n)"
        cache_mode_str = self.hdf5_cache_mode if self.hdf5_cache_mode is not None else "none"
        goal_mode_str = self.goal_mode if self.goal_mode is not None else "none"
        msg = msg.format(
            self.hdf5_path, self.obs_keys, self.seq_length,
            self.n_frame_stack, self.pad_seq_length, self.pad_frame_stack,
            goal_mode_str, cache_mode_str, self.k, self.n_trajs, self.total_num_sequences
        )
        return msg


class SequenceVLDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_dataset, task_emb):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return return_dict


def get_dataset(
    dataset_path,
    obs_modality,
    seq_len=1,
    frame_stack=1,
    traj_keys=None,
    hdf5_cache_mode="low_dim",
    add_reward=False,
    boundary_k=None,
    sample_step="auto",
    use_boundary=False,
    *args,
    **kwargs
):
    """
    Modified to support new flat trajectory format.
    obs_modality should be like: {"obs": ["main_images", "wrist_images", "robot_states"]}

    Returns:
        dataset: The dataset object
        task_name: Extracted task name from dataset path
    """
    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list

    dataset_keys = ["actions", "dones"]
    if add_reward:
        dataset_keys.append("rewards")

    dataset_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=all_obs_keys,
        dataset_keys=dataset_keys,
        load_next_obs=False,  
        frame_stack=frame_stack,
        seq_length=seq_len,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,
        hdf5_use_swmr=False,
        traj_keys=traj_keys,
        sample_step=sample_step,
        k=boundary_k,
    )

    if use_boundary:
        dataset = BoundarySequenceDataset(**dataset_kwargs)
    else:
        dataset = SequenceDataset(**dataset_kwargs)

    # Extract task name from dataset path
    filename = os.path.basename(dataset_path)
    # Remove file extension
    task_name = os.path.splitext(filename)[0]
    # Replace underscores with spaces
    task_name = task_name.replace("_", " ")

    return dataset, task_name