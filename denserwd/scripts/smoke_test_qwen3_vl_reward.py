import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

from denserwd.models.qwen3_vl import Qwen3VLReward


PARQUET_PATH = Path('/workspace/RLinf/denserwd/data/datasets/lerobot_libero/data/chunk-000/episode_000000.parquet')
TASKS_PATH = Path('/workspace/RLinf/denserwd/data/datasets/lerobot_libero/meta/tasks.jsonl')
MODEL_PATH = '/workspace/pretrained_vlas/Qwen3-VL-2B-Instruct'
SAVE_PATH = Path('/tmp/qwen3_vl_reward_smoke_save')
CONTEXT_WINDOW = 2


def load_task_map(path: Path) -> dict[int, str]:
    task_map = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        task_map[row['task_index']] = row['task']
    return task_map


def decode_image(image_dict: dict) -> Image.Image:
    return Image.open(io.BytesIO(image_dict['bytes'])).convert('RGB')


def main() -> None:
    table = pq.read_table(PARQUET_PATH)
    rows = table.to_pylist()[:CONTEXT_WINDOW]
    task_map = load_task_map(TASKS_PATH)

    task_prompt = task_map[rows[0]['task_index']]
    main_images = [decode_image(row['image']) for row in rows]
    wrist_images = [decode_image(row['wrist_image']) for row in rows]

    model = Qwen3VLReward(model_name_or_path=MODEL_PATH, torch_dtype=torch.bfloat16, device_map='auto')
    model.eval()

    with torch.no_grad():
        reward = model.predict_reward(task_prompt=task_prompt, main_images=main_images, wrist_images=wrist_images)

    print('task_prompt:', task_prompt)
    print('reward:', reward.detach().float().cpu().tolist())

    model.save_pretrained(SAVE_PATH)
    print('saved_to:', SAVE_PATH)
    print('saved_files:', sorted(str(p.relative_to(SAVE_PATH)) for p in SAVE_PATH.glob('**/*') if p.is_file())[:20])


if __name__ == '__main__':
    main()
