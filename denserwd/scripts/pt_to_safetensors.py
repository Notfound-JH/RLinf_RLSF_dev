# import torch
# from safetensors.torch import save_file

# # 加载原始模型权重（例如 .pth 文件）
# path = '/workspace/pretrained_vlas/openpi/torch/pi05_libero_few_shot/model.pt'
# state_dict = torch.load(path, map_location="cpu", weights_only=True)

# # 保存为 safetensors 格式
# target_path = '/workspace/pretrained_vlas/openpi/torch/pi05_libero_few_shot/model.safetensors'
# save_file(state_dict, target_path)

from safetensors.torch import load_file, save_file

state_dict = load_file("/workspace/pretrained_vlas/openpi/torch/pi05_libero_few_shot/model.safetensors")
cleaned = {k: v for k, v in state_dict.items() if not k.startswith("value_head.")}
save_file(cleaned, "clean_model5.safetensors")