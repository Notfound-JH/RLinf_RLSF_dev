#!/bin/bash
# Resume from latest ckpt (see libero_10_grpo_openpi_pi05_resume.yaml; currently global_step_20).
# 覆盖示例：GPUS=2,5,7 RAY_NUM_CPUS=32 RAY_PORT=29500 bash resume_tmux_binary.sh
GPUS=${GPUS:-2,5,7}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}
RAY_PORT=${RAY_PORT:-29500}
RAY_MEMORY_USAGE_THRESHOLD=${RAY_MEMORY_USAGE_THRESHOLD:-0.98}
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PUBLIC_DATA_DIR=${PUBLIC_DATA_DIR:-/mnt/storage/public_data}
DOCKER_IMAGE=${DOCKER_IMAGE:-docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero}

CKPT_ACTOR_DIR="/workspace/RLinf/models_ckpts/global_step_1500/actor"
NEW_FORMAT_DIR="${CKPT_ACTOR_DIR}/new_format_pytorch"

mkdir -p "${REPO_DIR}/logs"

tmux new-session -d -s rlinf_resume "cd \"${REPO_DIR}\" && docker rm -f rlinf 2>/dev/null; docker run --rm --gpus '\"device=${GPUS}\"' -e NVIDIA_VISIBLE_DEVICES=${GPUS} -e RAY_NUM_CPUS=${RAY_NUM_CPUS} -e RAY_PORT=${RAY_PORT} -e RAY_memory_usage_threshold=${RAY_MEMORY_USAGE_THRESHOLD} -e RAY_DISABLE_IMPORT_WARNING=1 --shm-size 100g --network host --name rlinf -v \"${REPO_DIR}\":/workspace/RLinf -v \"${PUBLIC_DATA_DIR}\":/mnt/storage/public_data \"${DOCKER_IMAGE}\" bash -c '
set -e
cd /workspace/RLinf/denserwd && source switch_env openpi
export PYTHONPATH=/workspace/RLinf:\$PYTHONPATH
export RAY_memory_usage_threshold=\${RAY_memory_usage_threshold}

if [ ! -f \"${NEW_FORMAT_DIR}/model.safetensors\" ]; then
  echo \"[convert] converting BC full-shot checkpoint (openpi_pytorch) to new-format...\"
  python /workspace/RLinf/denserwd/config/vla/convert_sft2new_pi05_bc_fullshot_gs1500.py
else
  echo \"[convert] new-format checkpoint already exists at ${NEW_FORMAT_DIR}, skip conversion\"
fi

ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray/* 2>/dev/null || true
ray start --head \\
  --port=\${RAY_PORT} \\
  --num-cpus=\${RAY_NUM_CPUS} \\
  --disable-usage-stats \\
  --include-dashboard=false
ray status

bash run_embodiment.sh libero_10_grpo_openpi_pi05_resume 2>&1 | tee -a /workspace/RLinf/logs/rlinf_resume_docker.log
'"
