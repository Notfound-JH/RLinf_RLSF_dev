import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer


class T5LanguageEncoder(nn.Module):
    """T5 language encoder for encoding text instructions."""

    def __init__(self, model_name: str = "t5-base", finetune: bool = False, device: torch.device='cuda', local_files_only: bool = False):
        super().__init__()
        self.finetune = finetune
        self.model = T5EncoderModel.from_pretrained(model_name, local_files_only=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.output_dim = self.model.config.d_model
        self.device = device

        if not finetune:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, texts: list, max_length=32) -> torch.Tensor:
        """Encode text inputs to embeddings.

        Args:
            texts: List of text strings

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        inputs = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state

        if not self.finetune:
            embeddings = embeddings.detach()

        return embeddings