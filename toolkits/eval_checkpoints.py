# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batch-evaluate saved embodied-agent checkpoints and pick the best one.

Workflow
--------
Phase 1 (fast scan, no video):
    For every checkpoint under ``--ckpt-root`` (``global_step_N``), run the
    standard eval entry point (``denserwd/eval_embodied_agent.py``) with
    video saving disabled, parse the aggregated ``eval/success_rate`` (and a
    few other metrics) from stdout, and print a ranked summary table.

Phase 2 (small confirmation run, with video):
    Re-run the single best checkpoint (by ``success_rate``) with a much
    smaller ``total_num_envs`` and video saving enabled, so you can visually
    inspect a handful of rollouts without paying the cost of recording video
    for every checkpoint.

Usage
-----
    python toolkits/eval_checkpoints.py \\
        --ckpt-root results/libero_10_grpo_openpi_pi05/checkpoints \\
        --config-name libero_10_grpo_openpi05_eval

Add ``--dry-run`` to just print the commands that would be executed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_SCRIPT = REPO_ROOT / "denserwd" / "eval_embodied_agent.py"
DEFAULT_CONFIG_PATH = REPO_ROOT / "denserwd" / "config" / "vla"

# Metrics we care about, in priority order for ranking.
RANK_METRIC = "success_rate"
METRICS_OF_INTEREST = [
    "success_rate",
    "success_once",
    "return",
    "normalized_reward_mean",
    "reward",
]

# Matches the dict that EmbodiedEvalRunner.run() logs, e.g.:
#   {'eval/success_rate': 0.42, 'eval/return': 0.3, ...}
EVAL_DICT_RE = re.compile(r"\{[^{}]*'eval/[^{}]*\}")

# Fallback: per-episode running tally printed by EnvWorker.env_evaluate_step,
# e.g. "... [23/64 succeeded so far]"
RUNNING_TALLY_RE = re.compile(r"\[(\d+)/(\d+) succeeded so far\]")


@dataclass
class CheckpointResult:
    step: int
    ckpt_path: Path
    log_path: Path
    metrics: dict = field(default_factory=dict)
    ok: bool = False
    error: str | None = None


def find_checkpoints(ckpt_root: Path) -> list[tuple[int, Path]]:
    """Discover ``global_step_N/actor/model_state_dict/full_weights.pt`` files."""
    results = []
    for step_dir in sorted(ckpt_root.glob("global_step_*")):
        m = re.match(r"global_step_(\d+)$", step_dir.name)
        if not m:
            continue
        step = int(m.group(1))
        weight_path = step_dir / "actor" / "model_state_dict" / "full_weights.pt"
        if weight_path.is_file():
            results.append((step, weight_path))
        else:
            print(f"[warn] skip {step_dir}: no full_weights.pt found", file=sys.stderr)
    return sorted(results, key=lambda x: x[0])


def build_command(
    eval_script: Path,
    config_path: Path,
    config_name: str,
    ckpt_path: Path,
    experiment_name: str,
    save_video: bool,
    total_num_envs: int,
    extra_overrides: list[str],
) -> list[str]:
    overrides = [
        f"runner.ckpt_path={ckpt_path}",
        f"runner.logger.experiment_name={experiment_name}",
        f"env.eval.video_cfg.save_video={'True' if save_video else 'False'}",
        f"env.eval.total_num_envs={total_num_envs}",
    ]
    overrides.extend(extra_overrides)
    cmd = [
        sys.executable,
        str(eval_script),
        "--config-path",
        str(config_path),
        "--config-name",
        config_name,
        *overrides,
    ]
    return cmd


def run_eval(
    cmd: list[str],
    log_path: Path,
    dry_run: bool,
) -> tuple[bool, str | None]:
    """Run one eval subprocess, teeing stdout/stderr into ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cmd] {' '.join(cmd)}")
    if dry_run:
        return True, None

    with open(log_path, "w") as f:
        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
    if proc.returncode != 0:
        return False, f"exit code {proc.returncode}, see {log_path}"
    return True, None


def parse_metrics(log_path: Path) -> dict:
    """Extract eval metrics from a log file, preferring the aggregated dict."""
    if not log_path.is_file():
        return {}

    text = log_path.read_text(errors="ignore")

    # Prefer the final aggregated metrics dict logged by EmbodiedEvalRunner.
    matches = EVAL_DICT_RE.findall(text)
    if matches:
        last = matches[-1]
        try:
            raw = ast.literal_eval(last)
        except (ValueError, SyntaxError):
            raw = {}
        metrics = {}
        for k, v in raw.items():
            key = k.split("/", 1)[-1]
            try:
                metrics[key] = float(v)
            except (TypeError, ValueError):
                pass
        if metrics:
            return metrics

    # Fallback: derive success_rate from the last "[X/Y succeeded so far]" line.
    tallies = RUNNING_TALLY_RE.findall(text)
    if tallies:
        succ, total = tallies[-1]
        succ, total = int(succ), int(total)
        if total > 0:
            return {"success_rate": succ / total}

    return {}


def print_summary(results: list[CheckpointResult]) -> None:
    header = ["step"] + METRICS_OF_INTEREST + ["status"]
    rows = []
    for r in results:
        row = [str(r.step)]
        for m in METRICS_OF_INTEREST:
            val = r.metrics.get(m)
            row.append(f"{val:.4f}" if isinstance(val, float) else "-")
        row.append("ok" if r.ok else f"FAILED ({r.error})")
        rows.append(row)

    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]
    line = " | ".join(h.ljust(w) for h, w in zip(header, widths))
    sep = "-+-".join("-" * w for w in widths)
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))


def pick_best(results: list[CheckpointResult]) -> CheckpointResult | None:
    candidates = [r for r in results if r.ok and RANK_METRIC in r.metrics]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.metrics[RANK_METRIC])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=REPO_ROOT / "results" / "libero_10_grpo_openpi_pi05" / "checkpoints",
        help="Directory containing global_step_N subfolders.",
    )
    parser.add_argument(
        "--config-name",
        default="libero_10_grpo_openpi_pi05",
        help="Hydra config name (under --config-path) used for evaluation. "
        "Defaults to the ORIGINAL TRAINING config (not a separate eval-only "
        "template), because eval_embodied_agent.py only launches the "
        "rollout/env worker groups (no actor), so cluster.component_placement's "
        "'actor' entry is irrelevant, while 'env,rollout' already maps to "
        "hardware rank 0 -- matching a single-GPU container out of the box. "
        "Using the training config also guarantees the eval-time model "
        "architecture/hyperparameters (num_action_chunks, model_path, etc.) "
        "exactly match how the checkpoint was trained.",
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated list of steps to evaluate, e.g. '10,20,30'. "
        "Default: auto-discover all checkpoints under --ckpt-root.",
    )
    parser.add_argument(
        "--eval-envs",
        type=int,
        default=50,
        help="total_num_envs used for the phase-1 full scan (no video).",
    )
    parser.add_argument(
        "--video-envs",
        type=int,
        default=5,
        help="total_num_envs used for the phase-2 confirmation run (with video).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "eval_checkpoints",
        help="Where to store per-run stdout logs and the summary json.",
    )
    parser.add_argument(
        "--skip-video-run",
        action="store_true",
        help="Only do phase 1 (metric scan); skip the phase-2 video run.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Extra Hydra override(s) to append to every run, "
        "e.g. --extra-override env.eval.task_suite_name=libero_10 (repeatable).",
    )
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="Add 'cluster.component_placement.actor=0' automatically. Needed "
        "when running inside a container/process that only exposes ONE GPU "
        "(e.g. `docker run --gpus device=N`), because the training config's "
        "default 'actor: 1-2' placement fails Hydra validation on a "
        "single-GPU host even though eval_embodied_agent.py never actually "
        "launches an actor worker group.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them.",
    )
    args = parser.parse_args()

    all_ckpts = find_checkpoints(args.ckpt_root)
    if not all_ckpts:
        print(f"[error] no checkpoints found under {args.ckpt_root}", file=sys.stderr)
        sys.exit(1)

    if args.steps:
        wanted = {int(s) for s in args.steps.split(",")}
        all_ckpts = [(s, p) for s, p in all_ckpts if s in wanted]
        missing = wanted - {s for s, _ in all_ckpts}
        if missing:
            print(f"[warn] requested steps not found: {sorted(missing)}", file=sys.stderr)

    args.log_dir.mkdir(parents=True, exist_ok=True)

    extra_overrides = list(args.extra_override)
    if args.single_gpu:
        extra_overrides.append("cluster.component_placement.actor=0")

    # ---- Phase 1: fast scan over all checkpoints, no video ----
    results: list[CheckpointResult] = []
    for step, ckpt_path in all_ckpts:
        exp_name = f"eval_scan_step_{step}"
        log_path = args.log_dir / f"{exp_name}.log"
        cmd = build_command(
            eval_script=args.eval_script,
            config_path=args.config_path,
            config_name=args.config_name,
            ckpt_path=ckpt_path,
            experiment_name=exp_name,
            save_video=False,
            total_num_envs=args.eval_envs,
            extra_overrides=extra_overrides,
        )
        ok, err = run_eval(cmd, log_path, args.dry_run)
        metrics = {} if args.dry_run else parse_metrics(log_path)
        results.append(
            CheckpointResult(
                step=step,
                ckpt_path=ckpt_path,
                log_path=log_path,
                metrics=metrics,
                ok=ok and (bool(metrics) or args.dry_run),
                error=err if not ok else (None if metrics or args.dry_run else "no metrics parsed"),
            )
        )

    print("\n=== Phase 1 summary (no video) ===")
    print_summary(sorted(results, key=lambda r: r.step))

    summary_path = args.log_dir / "eval_summary.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    "step": r.step,
                    "ckpt_path": str(r.ckpt_path),
                    "log_path": str(r.log_path),
                    "metrics": r.metrics,
                    "ok": r.ok,
                    "error": r.error,
                }
                for r in results
            ],
            indent=2,
        )
    )
    print(f"\n[info] summary written to {summary_path}")

    if args.dry_run:
        print("\n[dry-run] skipping phase 2 (best-checkpoint video run).")
        return

    best = pick_best(results)
    if best is None:
        print("[error] could not determine a best checkpoint (no valid metrics).")
        sys.exit(1)

    print(
        f"\n[info] best checkpoint: global_step_{best.step} "
        f"({RANK_METRIC}={best.metrics.get(RANK_METRIC):.4f})"
    )

    if args.skip_video_run:
        print("[info] --skip-video-run set, done.")
        return

    # ---- Phase 2: small confirmation run with video, best checkpoint only ----
    video_exp_name = f"eval_best_step_{best.step}_video"
    video_log_path = args.log_dir / f"{video_exp_name}.log"
    video_base_dir = args.log_dir / "best_ckpt_video"
    cmd = build_command(
        eval_script=args.eval_script,
        config_path=args.config_path,
        config_name=args.config_name,
        ckpt_path=best.ckpt_path,
        experiment_name=video_exp_name,
        save_video=True,
        total_num_envs=args.video_envs,
        extra_overrides=[
            f"env.eval.video_cfg.video_base_dir={video_base_dir}",
            *extra_overrides,
        ],
    )
    ok, err = run_eval(cmd, video_log_path, args.dry_run)
    video_metrics = parse_metrics(video_log_path)
    print("\n=== Phase 2: best-checkpoint confirmation run (with video) ===")
    if ok:
        print(f"[info] video saved under: {video_base_dir}")
        print(f"[info] metrics: {video_metrics}")
    else:
        print(f"[error] video run failed: {err}")


if __name__ == "__main__":
    main()
