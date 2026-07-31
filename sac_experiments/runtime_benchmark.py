"""Matched CPU/CUDA runtime benchmark for lightweight Go2 training hosts."""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from threading import Event, Thread
from time import perf_counter, sleep
from types import MappingProxyType
from typing import Any

import psutil
import torch

from sac_experiments.config import ExperimentConfig
from sac_experiments.env_utils import configure_torch, make_vec_env
from sac_experiments.model_factory import build_model


GIB = 1024**3
PROFILE_VERSION = 1


def _config_sha256(config: ExperimentConfig) -> str:
    """Fingerprint the exact YAML used to create a runtime profile."""
    return hashlib.sha256(config.config_path.read_bytes()).hexdigest()


def _host_info() -> dict[str, Any]:
    """Return stable host fields used to prevent cross-machine profile reuse."""
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


class ResourceSampler:
    """Sample process-tree memory and NVIDIA telemetry during one run."""

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._stop = Event()
        self._thread = Thread(target=self._sample_loop, daemon=True)
        self._started = False
        self.max_rss_bytes = 0
        self.swap_baseline_bytes = self._swap_used_bytes()
        self.max_swap_used_bytes = self.swap_baseline_bytes
        self.max_gpu_memory_mib: float | None = None
        self.max_gpu_temperature_c: float | None = None
        self.max_gpu_utilization_pct: float | None = None

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            processes = [self._process, *self._process.children(recursive=True)]
            rss = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.max_rss_bytes = max(self.max_rss_bytes, rss)
            swap_used = self._swap_used_bytes()
            if swap_used is not None:
                self.max_swap_used_bytes = max(
                    self.max_swap_used_bytes or 0,
                    swap_used,
                )
            self._sample_nvidia()
            sleep(0.25)

    @staticmethod
    def _swap_used_bytes() -> int | None:
        try:
            return int(psutil.swap_memory().used)
        except (OSError, NotImplementedError):
            return None

    def _sample_nvidia(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return
        if result.returncode != 0 or not result.stdout.strip():
            return
        try:
            memory, temperature, utilization = (
                float(item.strip())
                for item in result.stdout.splitlines()[0].split(",")
            )
        except ValueError:
            return
        self.max_gpu_memory_mib = max(self.max_gpu_memory_mib or 0.0, memory)
        self.max_gpu_temperature_c = max(
            self.max_gpu_temperature_c or 0.0,
            temperature,
        )
        self.max_gpu_utilization_pct = max(
            self.max_gpu_utilization_pct or 0.0,
            utilization,
        )


def _benchmark_config(
    config: ExperimentConfig,
    *,
    n_envs: int,
    device: str,
    transitions: int,
) -> ExperimentConfig:
    if 2048 % n_envs != 0:
        raise ValueError(f"n_envs={n_envs} must divide the 2048-transition rollout.")
    n_steps = 2048 // n_envs
    if transitions % 2048 != 0:
        raise ValueError("benchmark transitions must be divisible by 2048.")
    ppo = dict(config.ppo)
    ppo["n_steps"] = n_steps
    return replace(
        config,
        n_envs=n_envs,
        timesteps=transitions,
        device=device,
        allow_cpu=False,
        progress_bar=False,
        ppo=MappingProxyType(ppo),
    )


def apply_runtime_profile(
    config: ExperimentConfig,
    profile_path: Path,
) -> ExperimentConfig:
    """Apply a same-config, same-host PPO profile and revalidate constraints."""
    if config.algorithm != "PPO":
        raise ValueError("Runtime profiles can only be applied to PPO experiments.")
    report = json.loads(profile_path.read_text(encoding="utf-8"))
    if report.get("profile_version") != PROFILE_VERSION:
        raise ValueError("Runtime profile version is missing or unsupported.")
    if report.get("config_sha256") != _config_sha256(config):
        raise ValueError(
            "Runtime profile was produced from a different YAML configuration."
        )
    expected_host = _host_info()
    recorded_host = report.get("host")
    if not isinstance(recorded_host, dict) or any(
        recorded_host.get(key) != value
        for key, value in expected_host.items()
    ):
        raise ValueError(
            "Runtime profile belongs to a different host or Python/CUDA runtime."
        )
    selected = report.get("selected")
    if not isinstance(selected, dict):
        raise ValueError(f"Runtime profile has no selected candidate: {profile_path}")
    n_envs = int(selected["n_envs"])
    n_steps = int(selected["n_steps"])
    device = str(selected["device"])
    matching_runs = [
        run for run in report.get("runs", [])
        if isinstance(run, dict)
        and run.get("n_envs") == n_envs
        and run.get("n_steps") == n_steps
        and run.get("device") == device
        and run.get("status") == "completed"
        and run.get("eligible") is True
    ]
    if not matching_runs:
        raise ValueError("Selected runtime candidate is not a completed eligible run.")
    if n_envs not in {8, 16, 32} or device not in {"cpu", "cuda"}:
        raise ValueError("Runtime profile contains an unsupported candidate.")
    if n_envs * n_steps != 2048:
        raise ValueError("Runtime profile does not preserve a 2048-transition rollout.")
    if config.timesteps % (n_envs * n_steps) != 0:
        raise ValueError("Training timesteps must be divisible by the selected rollout.")
    if config.eval_freq % n_envs != 0:
        raise ValueError(
            "evaluation.frequency must be divisible by the selected n_envs."
        )
    ppo = dict(config.ppo)
    batch_size = int(ppo["batch_size"])
    if batch_size > n_envs * n_steps or (n_envs * n_steps) % batch_size != 0:
        raise ValueError("Selected rollout is incompatible with PPO batch_size.")
    ppo["n_steps"] = n_steps
    resolved = replace(
        config,
        n_envs=n_envs,
        device=device,
        allow_cpu=False,
        ppo=MappingProxyType(ppo),
    )
    print(
        "Applied validated runtime profile: "
        f"{n_envs} envs, n_steps={n_steps}, device={device}. "
        "These values are part of the recorded experiment design."
    )
    return resolved


def run_runtime_benchmark(
    config: ExperimentConfig,
    *,
    transitions: int = 131_072,
    output_path: Path = Path("outputs/go2_runtime_benchmark.json"),
) -> Path:
    """Benchmark 8/16/32 envs on CPU/CUDA and select a safe profile."""
    if config.algorithm != "PPO":
        raise ValueError("Runtime benchmark currently supports PPO configs only.")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    results: list[dict[str, Any]] = []
    for n_envs in (8, 16, 32):
        for requested_device in ("cpu", "cuda"):
            if requested_device == "cuda" and not torch.cuda.is_available():
                results.append({
                    "n_envs": n_envs,
                    "device": requested_device,
                    "status": "skipped",
                    "reason": "CUDA is not available",
                })
                continue

            run_config = _benchmark_config(
                config,
                n_envs=n_envs,
                device=requested_device,
                transitions=transitions,
            )
            device = configure_torch(requested_device, allow_cpu=False)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            sampler = ResourceSampler()
            try:
                with closing(make_vec_env(
                    run_config.env_id,
                    n_envs=n_envs,
                    seed=run_config.seed,
                    env_kwargs=run_config.env_kwargs,
                )) as env:
                    model = build_model(
                        run_config,
                        env,
                        device=device,
                        tensorboard_log=None,
                    )
                    sampler.start()
                    started = perf_counter()
                    model.learn(
                        total_timesteps=transitions,
                        log_interval=None,
                        progress_bar=False,
                    )
                    wall_time = perf_counter() - started
            except Exception as exc:
                results.append({
                    "n_envs": n_envs,
                    "device": requested_device,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                continue
            finally:
                sampler.stop()

            swap_delta = (
                max(
                    0,
                    sampler.max_swap_used_bytes - sampler.swap_baseline_bytes,
                )
                if (
                    sampler.max_swap_used_bytes is not None
                    and sampler.swap_baseline_bytes is not None
                )
                else None
            )
            torch_vram = (
                torch.cuda.max_memory_allocated()
                if device == "cuda"
                else 0
            )
            max_vram_bytes = max(
                torch_vram,
                int((sampler.max_gpu_memory_mib or 0.0) * 1024**2),
            )
            eligible = (
                sampler.max_rss_bytes <= 12 * GIB
                and (
                    swap_delta is None
                    or swap_delta <= 512 * 1024**2
                )
                and max_vram_bytes <= 6 * GIB
            )
            results.append({
                "n_envs": n_envs,
                "n_steps": 2048 // n_envs,
                "device": requested_device,
                "status": "completed",
                "eligible": eligible,
                "transitions": transitions,
                "wall_time_seconds": wall_time,
                "transitions_per_second": transitions / wall_time,
                "max_rss_bytes": sampler.max_rss_bytes,
                "swap_delta_bytes": swap_delta,
                "swap_sampling_available": swap_delta is not None,
                "max_gpu_memory_bytes": max_vram_bytes,
                "max_gpu_temperature_c": sampler.max_gpu_temperature_c,
                "max_gpu_utilization_pct": sampler.max_gpu_utilization_pct,
            })

    eligible = [
        result for result in results
        if result.get("status") == "completed" and result.get("eligible")
    ]
    eligible.sort(
        key=lambda item: (
            -item["transitions_per_second"],
            item["max_rss_bytes"],
            item["n_envs"],
        )
    )
    selected = eligible[0] if eligible else None
    if selected and len(eligible) > 1:
        fastest = selected["transitions_per_second"]
        near_ties = [
            result for result in eligible
            if result["transitions_per_second"] >= 0.9 * fastest
        ]
        selected = min(
            near_ties,
            key=lambda item: (item["max_rss_bytes"], item["n_envs"]),
        )

    report = {
        "profile_version": PROFILE_VERSION,
        "config": str(config.config_path),
        "config_sha256": _config_sha256(config),
        "host": _host_info(),
        "constraints": {
            "max_rss_bytes": 12 * GIB,
            "max_swap_delta_bytes": 512 * 1024**2,
            "max_gpu_memory_bytes": 6 * GIB,
        },
        "runs": results,
        "selected": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Runtime benchmark report: {output_path}")
    if selected:
        print(
            "Selected "
            f"{selected['n_envs']} envs / {selected['device']} "
            f"({selected['transitions_per_second']:.1f} transitions/s)"
        )
    else:
        print("No runtime profile met the memory constraints.")
    return output_path
