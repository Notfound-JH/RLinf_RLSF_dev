---
license: cc-by-4.0
task_categories:
- robotics
tags:
- LeRobot
- libero
- panda
- rlds
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description

This dataset combines four individual Libero datasets: Libero-Spatial, Libero-Object, Libero-Goal and Libero-10. 
All datasets were taken from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds) and converted into LeRobot format.


- **Homepage:** https://libero-project.github.io
- **Paper:** https://arxiv.org/abs/2306.03310
- **License:** CC-BY 4.0

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v2.0",
    "robot_type": "panda",
    "total_episodes": 1693,
    "total_frames": 273465,
    "total_tasks": 40,
    "total_videos": 0,
    "total_chunks": 2,
    "chunks_size": 1000,
    "fps": 10,
    "splits": {
        "train": "0:1693"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "image": {
            "dtype": "image",
            "shape": [
                256,
                256,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ]
        },
        "wrist_image": {
            "dtype": "image",
            "shape": [
                256,
                256,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ]
        },
        "state": {
            "dtype": "float32",
            "shape": [
                8
            ],
            "names": [
                "state"
            ]
        },
        "actions": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": [
                "actions"
            ]
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    }
}
```


## Citation

**BibTeX:**

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```