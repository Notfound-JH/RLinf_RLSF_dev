"""One-off conversion of the BC full-shot LIBERO SFT checkpoint (global_step_1500,
trained with the new self-contained `openpi_pytorch` model_type) into the
new-format layout consumable by `rlinf.models.embodiment.openpi_pytorch.get_model`
(actor.model.model_type: openpi_pytorch).

Run inside the docker image with `openpi` installed (source switch_env openpi),
with PYTHONPATH including the repo root, e.g.:

    export PYTHONPATH=/workspace/RLinf:$PYTHONPATH
    python denserwd/config/vla/convert_sft2new_pi05_bc_fullshot_gs1500.py
"""

from rlinf.utils.ckpt_convertor.openpi import sft2new

CKPT_ACTOR_DIR = "/workspace/RLinf/models_ckpts/global_step_1500/actor"

# The consolidated full-weights file for this checkpoint lives under
# ema_model_state_dict/ (not the sft2new default model_state_dict/), so we pass
# the file path directly.
CKPT = f"{CKPT_ACTOR_DIR}/ema_model_state_dict/full_weights.pt"

# LIBERO norm stats bundled with the original RLinf-Pi05-LIBERO-SFT base
# checkpoint (asset_id = "physical-intelligence/libero", matching
# rlinf/models/embodiment/openpi/dataconfig's pi05_libero TrainConfig).
INPUT_NORM_STATS = (
    "/mnt/storage/public_data/models/RLinf-Pi05-LIBERO-SFT/"
    "physical-intelligence/libero/norm_stats.json"
)

OUTPUT_MODEL = f"{CKPT_ACTOR_DIR}/new_format_pytorch"
OUTPUT_NORM_STATS = f"{OUTPUT_MODEL}/physical-intelligence/libero/norm_stats.json"

if __name__ == "__main__":
    sft2new.convert(CKPT, INPUT_NORM_STATS, OUTPUT_MODEL, OUTPUT_NORM_STATS)
