from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="lerobot/pi0_base",
#     local_dir="/mnt/storage/public_data/models/openpi/torch/pi05_base",
#     local_dir_use_symlinks=False  # 可选：避免创建符号链接，直接复制文件
# )

# 指定你要下载的模型和目标路径
model_id = "Qwen/Qwen3-VL-2B-Instruct"
local_dir = "/mnt/storage/public_data/models/Qwen3-VL-2B-Instruc"  # ← 改成你自己的路径

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 避免创建软链接，直接复制文件（推荐）
    resume_download=True,          # 支持断点续传
    token=None,                    # 如果是私有模型，填你的 HF token
)