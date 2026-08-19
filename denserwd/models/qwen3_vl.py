import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_GOAL_PROMPT = (
    "You are a reward model for robotic manipulation. "
    "Given the task instruction and the ordered history of observations, "
    "predict the probability that the task has been successfully completed. "
    "Return a scalar reward in [0, 1], where 1 means success is very likely and 0 means failure is very likely. "
    "The multimodal inputs are ordered as: goal prompt, task prompt, main camera image history, wrist camera image history."
)


class Qwen3VLReward(nn.Module):
    """Qwen3-VL-based reward model for multimodal robotic trajectories."""

    def __init__(
        self,
        cfg=None,
        model_name_or_path: str = '/workspace/pretrained_vlas/Qwen3-VL-2B-Instruct',
        goal_prompt: str = DEFAULT_GOAL_PROMPT,
        torch_dtype: torch.dtype | str = 'auto',
        device_map: str | dict[str, Any] = 'auto',
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg is not None:
            model_cfg = getattr(cfg, 'model', cfg)
            model_name_or_path = getattr(model_cfg, 'pretrained_path', model_name_or_path)
            goal_prompt = getattr(model_cfg, 'goal_prompt', goal_prompt)
            torch_dtype = getattr(model_cfg, 'torch_dtype', torch_dtype)
            device_map = getattr(model_cfg, 'device_map', device_map)
            freeze_backbone = getattr(model_cfg, 'freeze_backbone', freeze_backbone)

        self.model_name_or_path = str(model_name_or_path)
        self.goal_prompt = goal_prompt
        self.task_prompts: list[str] | None = None
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        hidden_size = self.vlm.config.text_config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.reward_head.to(device=self.vlm.device, dtype=self.vlm.dtype)
        self.modules_to_save = nn.ModuleList([self.vlm, self.reward_head])

        if freeze_backbone:
            for param in self.vlm.parameters():
                param.requires_grad = False

    @property
    def device(self) -> torch.device:
        return self.vlm.device

    def get_task_embs(self, task_descriptions: list[str]) -> list[str]:
        return list(task_descriptions)

    def set_task_prompts(self, task_prompts: list[str]) -> None:
        self.task_prompts = list(task_prompts)

    def forward(
        self,
        data_or_goal_prompt,
        task_prompt: str | list[str] | None = None,
        main_images: list[Image.Image] | list[list[Image.Image]] | torch.Tensor | None = None,
        wrist_images: list[Image.Image] | list[list[Image.Image]] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(data_or_goal_prompt, dict):
            goal_prompt, task_prompt, main_images, wrist_images = self._extract_from_batch(data_or_goal_prompt)
        else:
            goal_prompt = data_or_goal_prompt

        messages = self._build_batch_messages(goal_prompt, task_prompt, main_images, wrist_images)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
            processor_kwargs={'padding': True},
        )
        inputs = inputs.to(self.device)
        outputs = self.vlm(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        last_hidden_state = outputs.hidden_states[-1]

        attention_mask = inputs.get('attention_mask')
        if attention_mask is None:
            pooled = last_hidden_state[:, -1, :]
        else:
            last_token_indices = attention_mask.sum(dim=1) - 1
            pooled = last_hidden_state[
                torch.arange(last_hidden_state.size(0), device=last_hidden_state.device),
                last_token_indices,
            ]

        pooled = pooled.to(dtype=next(self.reward_head.parameters()).dtype)
        rewards = self.reward_head(pooled).squeeze(-1)
        return rewards

    def predict(self, data: dict) -> torch.Tensor:
        return self.forward(data)

    def predict_reward(
        self,
        task_prompt: str | list[str],
        main_images: list[Image.Image] | list[list[Image.Image]] | torch.Tensor,
        wrist_images: list[Image.Image] | list[list[Image.Image]] | torch.Tensor,
        goal_prompt: str | None = None,
    ) -> torch.Tensor:
        return self.forward(goal_prompt or self.goal_prompt, task_prompt, main_images, wrist_images)

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        backbone_path = save_path / 'backbone'
        processor_path = save_path / 'processor'
        self.vlm.save_pretrained(backbone_path)
        self.processor.save_pretrained(processor_path)
        torch.save(self.reward_head.state_dict(), save_path / 'reward_head.pt')
        (save_path / 'reward_config.json').write_text(
            json.dumps(
                {
                    'goal_prompt': self.goal_prompt,
                    'model_name_or_path': self.model_name_or_path,
                    'task_prompts': self.task_prompts,
                },
                indent=2,
            )
        )

    @classmethod
    def from_pretrained_reward_model(
        cls,
        load_directory: str | Path,
        torch_dtype: torch.dtype | str = 'auto',
        device_map: str | dict[str, Any] = 'auto',
    ) -> 'Qwen3VLReward':
        load_path = Path(load_directory)
        cfg = json.loads((load_path / 'reward_config.json').read_text())
        model = cls(
            model_name_or_path=str(load_path / 'backbone'),
            goal_prompt=cfg['goal_prompt'],
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        model.processor = AutoProcessor.from_pretrained(load_path / 'processor')
        model.reward_head.load_state_dict(torch.load(load_path / 'reward_head.pt', map_location=model.device))
        model.reward_head.to(device=model.device, dtype=model.vlm.dtype)
        model.modules_to_save = nn.ModuleList([model.vlm, model.reward_head])
        model.model_name_or_path = cfg.get('model_name_or_path', str(load_path / 'backbone'))
        model.task_prompts = cfg.get('task_prompts')
        return model

    def _extract_from_batch(self, batch: dict):
        obs = batch.get('obs', batch)
        main_images = self._first_present(obs, ['main_images'])
        if main_images is None:
            main_images = self._first_present(batch, ['main_images', 'image'])
        wrist_images = self._first_present(obs, ['wrist_images'])
        if wrist_images is None:
            wrist_images = self._first_present(batch, ['wrist_images', 'wrist_image'])
        task_prompt = self._extract_task_prompt(batch)
        return self.goal_prompt, task_prompt, main_images, wrist_images


    def _first_present(self, data: dict, keys: list[str]):
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return None

    def _extract_task_prompt(self, batch: dict) -> str | list[str]:
        if 'task_prompt' in batch:
            return batch['task_prompt']
        if 'task_emb' in batch and self._is_text_like(batch['task_emb']):
            return batch['task_emb']
        if 'task_index' in batch and self.task_prompts is not None:
            task_indices = batch['task_index']
            if isinstance(task_indices, torch.Tensor):
                task_indices = task_indices.detach().cpu().tolist()
            if isinstance(task_indices, int):
                return self.task_prompts[task_indices]
            return [self.task_prompts[int(idx)] for idx in task_indices]
        raise ValueError('Unable to extract task prompt for Qwen3VLReward.')

    def _is_text_like(self, value: Any) -> bool:
        if isinstance(value, str):
            return True
        return isinstance(value, list) and (len(value) == 0 or isinstance(value[0], str))

    def _build_batch_messages(
        self,
        goal_prompt: str,
        task_prompt: str | list[str],
        main_images: list[Image.Image] | list[list[Image.Image]] | torch.Tensor,
        wrist_images: list[Image.Image] | list[list[Image.Image]] | torch.Tensor,
    ) -> list[list[dict[str, Any]]]:
        task_prompts = [task_prompt] if isinstance(task_prompt, str) else task_prompt
        main_image_batches = self._normalize_image_batches(main_images, len(task_prompts))
        wrist_image_batches = self._normalize_image_batches(wrist_images, len(task_prompts))

        messages = []
        for prompt, main_batch, wrist_batch in zip(task_prompts, main_image_batches, wrist_image_batches):
            content: list[dict[str, Any]] = [
                {'type': 'text', 'text': goal_prompt},
                {'type': 'text', 'text': f'Task prompt: {prompt}'},
            ]
            content.extend({'type': 'image', 'image': image} for image in main_batch)
            content.extend({'type': 'image', 'image': image} for image in wrist_batch)
            messages.append([{'role': 'user', 'content': content}])
        return messages

    def _normalize_image_batches(
        self,
        images: list[Image.Image] | list[list[Image.Image]] | torch.Tensor,
        batch_size: int,
    ) -> list[list[Image.Image]]:
        if isinstance(images, torch.Tensor):
            if images.ndim == 4:
                images = images.unsqueeze(0)
            if images.ndim != 5:
                raise ValueError(f'Expected image tensor with 4 or 5 dims, got shape {tuple(images.shape)}')
            return [self._tensor_sequence_to_pil_list(sample) for sample in images]

        if batch_size == 1 and images and isinstance(images[0], Image.Image):
            return [[self._to_pil_image(image) for image in images]]

        return [
            [self._to_pil_image(image) for image in sample]
            for sample in images
        ]

    def _tensor_sequence_to_pil_list(self, tensor: torch.Tensor) -> list[Image.Image]:
        return [self._to_pil_image(image) for image in tensor]

    def _to_pil_image(self, image: Image.Image | torch.Tensor | np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert('RGB')

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)

        if image.ndim != 3:
            raise ValueError(f'Expected 3D image array, got shape {image.shape}')
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return Image.fromarray(image).convert('RGB')
