#!/usr/bin/env python
"""Train the point-conditioned diffusion policy (NoMaD hyperparameter base).

Loss = MSE(ε) + w·BCE(stop logit), AdamW + linear-warmup cosine schedule,
EMA weights are served. The stop head has a default loss weight of 0.5;
validation logs report precision and recall.
Optional classifier-free-guidance condition dropout (``--cfg-drop-prob``)
and coordinate-token conditioning (``--use-coord-embed``).

Checkpoint contract (read by NomadBackend unchanged): {"state_dict": EMA,
"raw_state_dict", "step", "train_config", "waypoint_scale"}. Run dirs are
append-only: an existing ckpt aborts the run unless --overwrite is given,
and a provenance sidecar (config + git commit + dataset manifest hash) is
written to <out>/config.json at start.

0236 run:
  python -m supernav.methods.localnav_policy.train \
      --data data/localnav/0236_v2/samples.jsonl --root data/localnav/0236_v2 \
      --out data/runs/localnav/exp02_stop_head --steps 15000 --batch 256 --amp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from supernav.paths import workspace_root

import numpy as np


import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from supernav.methods.localnav_policy.model import (  # noqa: E402
    ACTION_HORIZON,
    DEFAULT_WAYPOINT_SCALE_M,
    FLOW_TIME_SCALE,
    NomadPolicy,
)
from supernav.methods.localnav_policy.train_dataset import (  # noqa: E402
    LocalNavTorchDataset,
)


@dataclass
class TrainConfig:
    # One samples.jsonl + image root per source; parallel lists mix sources
    # (e.g. expert data + a DAgger round) at their natural proportions.
    data: "str | tuple"
    root: "str | tuple"
    out: str
    steps: int = 30000
    batch: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    warmup: int = 500
    ema_decay: float = 0.999
    stop_loss_weight: float = 0.5
    stop_pos_weight: float = 3.0
    stop_k: int = 2
    cfg_drop_prob: float = 0.0
    use_coord_embed: bool = False
    # Auxiliary distance regression regularizes the shared encoder.
    use_dist_head: bool = False
    dist_loss_weight: float = 0.0
    # Goal-point labels come from point_track.npz sidecars.
    use_point_head: bool = False
    point_loss_weight: float = 0.0
    # Offline advantage-weighted behavior cloning: on-policy
    # samples get eps-loss weights exp((return − baseline)/beta) computed
    # from their privileged true_remaining_m sequences (potential-based
    # progress reward + terminal stop bonus). 0.0 disables (= plain BC).
    rl_advantage_beta: float = 0.0
    amp: bool = False
    overwrite: bool = False
    val_fraction: float = 0.05
    # "hop": held-out whole hops (honest val). "sample": sliding-window
    # split — samples of one hop straddle train/val, inflating val metrics
    val_split: str = "hop"
    # Repeat stop-POSITIVE on-policy samples (source=localnav_dagger) this many
    # times in the train split.
    onpolicy_positive_repeat: int = 1
    far_sample_weight: float = 1.0
    sim2real_aug: float = 0.0
    far_sample_threshold: int = 16
    source_weights: tuple = ()
    context_mode: str = "fixed"
    memory_budget: int = 12
    memory_recent_tail: int = 0
    memory_anchor: bool = True
    memory_mid_span: int = 0
    memory_mid_count: int = 0
    hindsight_action_weight: float = 1.0
    device: str = "cuda"
    workers: int = 4
    point_jitter_px: float = 2.0
    waypoint_scale: float = DEFAULT_WAYPOINT_SCALE_M
    # Planning horizon in meters is action_horizon times waypoint_scale.
    action_horizon: int = ACTION_HORIZON
    zero_motion_stop: bool = False
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 50
    save_every: int = 1000
    # Rank-0 validation subsample cap (0 = no cap); keeps the barrier wait
    # bounded on multi-million-sample mixes.
    max_val_samples: int = 20000
    # ---- M3 ablation knobs (defaults reproduce the v1 architecture; every
    # knob is stored in the ckpt's train_config so serving self-configures) --
    encoder: str = "effnet-scratch"  # effnet-scratch | effnet-imagenet | dino-s-frozen
    image_size: int = 96             # policy input side (dino needs %14==0)
    head_type: str = "ddpm"          # ddpm | flow (rectified-flow velocity head)
    conditioning: str = "marker"     # marker | crop | coord
    flow_infer_steps: int = 4        # Euler steps at sampling (flow head only)
    context_size: int = 4            # observation memory window (frames)


def _attach_point_track(samples, root_path, is_main: bool) -> None:
    """Attach point_track.npz labels by sample_id using sorted sidecar IDs."""
    import numpy as np

    sidecar = os.path.join(os.fspath(root_path), "point_track.npz")
    if not os.path.isfile(sidecar):
        raise FileNotFoundError(
            f"--use-point-head requires {sidecar}; run precompute_point_track.py first"
        )
    data = np.load(sidecar)
    ids, uvv = data["ids"], data["uvv"]
    want = np.asarray([s.sample_id for s in samples], dtype=ids.dtype)
    pos = np.searchsorted(ids, want)
    pos_clipped = np.clip(pos, 0, len(ids) - 1)
    hit = ids[pos_clipped] == want
    miss = int((~hit).sum())
    if miss:
        raise ValueError(
            f"{sidecar} is missing {miss}/{len(want)} sample IDs; sidecar and dataset root are out of sync"
        )
    rows = uvv[pos_clipped]
    for s, row in zip(samples, rows):
        # The frozen sample holds a mutable metadata dictionary.
        if s.metadata is None:
            raise ValueError(f"sample {s.sample_id} has no metadata; invalid dataset root")
        s.metadata["_point_track"] = (float(row[0]), float(row[1]), float(row[2]))
    if is_main:
        vis = float((rows[:, 2] > 0.5).mean())
        unknown = float((rows[:, 2] < 0.0).mean())
        print(
            f"[localnav-train] point-track labels: {os.path.basename(os.fspath(root_path))} "
            f"n={len(want):,} visible={vis:.2%} unlabeled={unknown:.2%}",
            flush=True,
        )


def stop_bce_loss(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    keep_mask: "torch.Tensor",
    pos_weight: "float | torch.Tensor" = 1.0,
) -> "torch.Tensor":
    """BCE over rows that still see the goal. Goal-dropped (CFG) rows carry no
    stop supervision — 'am I there yet' is meaningless without a there."""
    keep = keep_mask.to(torch.bool)
    if int(keep.sum()) == 0:
        return logits.sum() * 0.0
    if not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor(float(pos_weight), device=logits.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[keep], labels[keep], pos_weight=pos_weight
    )


def stop_precision_recall(
    probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> "tuple[float, float]":
    """The stop head's KPI pair (loss magnitude is not it)."""
    probs = np.asarray(probs, dtype=np.float64)
    positives = np.asarray(labels, dtype=np.float64) > 0.5
    fired = probs >= threshold
    true_positives = float(np.sum(fired & positives))
    precision = true_positives / fired.sum() if fired.any() else 0.0
    recall = true_positives / positives.sum() if positives.any() else 0.0
    return float(precision), float(recall)


def _git_commit() -> "str | None":
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace_root()), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):  # noqa: BLE001 — provenance is best-effort outside a checkout
        return None


def _manifest_sha256(root: str) -> "str | None":
    manifest = Path(root) / "manifest.json"
    if not manifest.is_file():
        return None
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def annotate_onpolicy_advantages(
    samples,
    beta: float,
    *,
    gamma: float = 0.9,
    weight_clip: "tuple[float, float]" = (0.2, 5.0),
) -> int:
    """RL-R1: set `_rl_weight` on on-policy samples from privileged rewards.

    Per rollout (episode), reward_t = geo_t − geo_{t+1} (potential-based
    progress; collisions/detours show up as negative progress) plus a
    terminal bonus (+2 ended ≤1m, +1 ≤2m, −0.5 far, extra −2 for a false
    'reached'). Discounted returns minus a distance-binned baseline feed
    w = clip(exp(A/beta)). Expert samples keep weight 1. Returns the number
    of annotated samples; beta<=0 is a no-op (plain BC)."""
    if beta <= 0.0:
        return 0
    episodes: dict = {}
    for sample in samples:
        meta = sample.metadata or {}
        if meta.get("dagger") and meta.get("true_remaining_m") is not None:
            episodes.setdefault(sample.episode_id, []).append(sample)

    bins = [0.0, 1.0, 2.0, 4.0, 8.0, float("inf")]
    per_bin: dict = {i: [] for i in range(len(bins) - 1)}
    returns: dict = {}
    for episode in episodes.values():
        episode.sort(key=lambda s: s.current_frame_index)
        geos = [float(s.metadata["true_remaining_m"]) for s in episode]
        rewards = [geos[i] - geos[i + 1] for i in range(len(geos) - 1)]
        terminal = 2.0 if geos[-1] <= 1.0 else (1.0 if geos[-1] <= 2.0 else -0.5)
        if episode[-1].metadata.get("rollout_status") == "reached" and geos[-1] > 2.0:
            terminal -= 2.0
        rewards.append(terminal)
        acc = 0.0
        for index in range(len(episode) - 1, -1, -1):
            acc = rewards[index] + gamma * acc
            returns[id(episode[index])] = acc
            geo = geos[index]
            for b in range(len(bins) - 1):
                if bins[b] <= geo < bins[b + 1]:
                    per_bin[b].append((geo, acc))
                    break

    baseline = {
        b: (float(np.mean([r for _, r in rows])) if rows else 0.0)
        for b, rows in per_bin.items()
    }

    def bin_of(geo: float) -> int:
        for b in range(len(bins) - 1):
            if bins[b] <= geo < bins[b + 1]:
                return b
        return len(bins) - 2

    annotated = 0
    low, high = weight_clip
    for episode in episodes.values():
        for sample in episode:
            geo = float(sample.metadata["true_remaining_m"])
            advantage = returns[id(sample)] - baseline[bin_of(geo)]
            sample.metadata["_rl_weight"] = float(
                np.clip(math.exp(advantage / beta), low, high)
            )
            annotated += 1
    return annotated


def oversample_onpolicy_positives(train_samples, repeat: int, stop_k: int):
    """Duplicate stop-positive on-policy samples (repeat-1) extra times.
    Only touches source='localnav_dagger' rows; expert data is untouched."""
    if repeat <= 1:
        return list(train_samples)
    positives = [
        s
        for s in train_samples
        if s.source == "localnav_dagger"
        and (s.target_frame_index - s.current_frame_index) <= stop_k
    ]
    return list(train_samples) + positives * (repeat - 1)


def oversample_far_targets(train_samples, weight: float, threshold: int, seed: int = 0):
    """Give far-target samples ``weight`` times the sampling mass of near ones.

    Exposure is allocated, not added: the sampler stays uniform over the returned
    list, so duplicating far rows raises their share of a fixed step budget and
    lowers everyone else's. Collecting a far-heavy pack instead needs roughly
    16x the samples that a 6%-of-mix pack delivers, which is why this knob exists.

    ``threshold`` is in remaining primitives (0.25 m or 10 deg each), so 16 is
    about 4 m of walking. The fractional part of ``weight`` is spread over a
    deterministic stride, keeping the result reproducible from ``seed``.
    """
    if weight <= 1.0 or threshold <= 0:
        return list(train_samples)
    far = [
        s
        for s in train_samples
        if (s.target_frame_index - s.current_frame_index) >= threshold
    ]
    if not far:
        return list(train_samples)
    extra_whole = int(weight) - 1
    out = list(train_samples) + far * extra_whole
    fraction = weight - int(weight)
    if fraction > 0:
        rng = random.Random(seed)
        out.extend(s for s in far if rng.random() < fraction)
    return out


def oversample_by_root(train_samples, root_path, specs, seed: int = 0):
    """Weight a whole data source by duplicating its rows.

    ``specs`` is a list of "substring=weight"; a root whose path contains the
    substring has its samples repeated to that weight. Same allocation logic as
    oversample_far_targets -- exposure is redistributed inside a fixed step
    budget, nothing new is collected.

    Motivation: clicking a raised surface instead of the floor costs 0.20-0.36
    success at matched target distance (measured on the object-point track), and
    that semantics is only 12 percent of the mix.
    """
    if not specs:
        return list(train_samples)
    root = str(root_path)
    weight = 1.0
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"source weight must be substring=weight, got {spec!r}")
        needle, value = spec.rsplit("=", 1)
        if needle and needle in root:
            weight = float(value)
    if weight <= 1.0:
        return list(train_samples)
    rows = list(train_samples)
    out = rows * int(weight)
    fraction = weight - int(weight)
    if fraction > 0:
        rng = random.Random(seed)
        out.extend(s for s in rows if rng.random() < fraction)
    return out


def split_train_val(samples, val_fraction: float, seed: int, mode: str = "hop"):
    """(val, train). mode='hop' holds out whole hops — samples of one hop must
    never straddle the split (the 'sample' mode leaks every val hop
    into train, inflating on-manifold val metrics)."""
    rng = random.Random(seed)
    if mode == "hop":
        episodes = sorted({s.episode_id for s in samples})
        rng.shuffle(episodes)
        val_episodes = set(episodes[: int(len(episodes) * val_fraction)])
        val = [s for s in samples if s.episode_id in val_episodes]
        train = [s for s in samples if s.episode_id not in val_episodes]
        rng.shuffle(train)
        return val, train
    if mode == "sample":
        pool = list(samples)
        rng.shuffle(pool)
        val_count = int(len(pool) * val_fraction)
        return pool[:val_count], pool[val_count:]
    raise ValueError(f"val_split must be 'hop' or 'sample', got {mode!r}")


class _Ema:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self._decay = float(decay)
        self._shadow = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, tensor in model.state_dict().items():
            shadow = self._shadow[name]
            if tensor.dtype.is_floating_point:
                shadow.mul_(self._decay).add_(tensor, alpha=1.0 - self._decay)
            else:
                shadow.copy_(tensor)

    def state_dict(self) -> dict:
        return {name: tensor.clone() for name, tensor in self._shadow.items()}


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _dist_env() -> "tuple[int, int, int]":
    """(rank, world_size, local_rank); (0, 1, 0) unless launched by torchrun."""
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def _gca_collate(items):
    """Pad variable-length contexts to the batch max; True in pad_mask marks
    padding slots (excluded from encoder, attention, and pooling)."""
    import torch as _t

    max_k = max(it["context"].shape[0] for it in items)
    batch = {}
    ctxs, ages, masks = [], [], []
    for it in items:
        k = it["context"].shape[0]
        pad = max_k - k
        ctx = it["context"]
        age = it["frame_ages"]
        if pad:
            ctx = _t.cat([ctx, ctx.new_zeros(pad, *ctx.shape[1:])], dim=0)
            age = _t.cat([age, age.new_zeros(pad)], dim=0)
        ctxs.append(ctx)
        ages.append(age)
        masks.append(
            _t.cat([_t.zeros(k, dtype=_t.bool), _t.ones(pad, dtype=_t.bool)])
        )
    batch["context"] = _t.stack(ctxs)
    batch["frame_ages"] = _t.stack(ages)
    batch["pad_mask"] = _t.stack(masks)
    for key in items[0]:
        if key in ("context", "frame_ages"):
            continue
        batch[key] = _t.stack([it[key] for it in items])
    return batch


def _infinite(loader: DataLoader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)  # reshuffle shards each pass
        yield from loader
        epoch += 1


def run_training(config: TrainConfig) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    rank, world_size, local_rank = _dist_env()
    distributed = world_size > 1
    device = torch.device(config.device)
    if distributed:
        import torch.distributed as dist

        # 1h collective timeout: rank-0 duties (validation / checkpointing)
        # scale with dataset size and must not abort the other ranks — the
        # default 10 min killed the first multiscene run mid-first-val.
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            timeout=timedelta(hours=1),
        )
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
    is_main = rank == 0
    if device.type == "cpu":
        # Guard against thread oversubscription on many-core shared boxes:
        # tiny-batch CPU training with default torch threading can thrash
        # (observed: 27 cores busy, slower than 4). GPU runs are unaffected.
        torch.set_num_threads(
            int(os.environ.get("LOCALNAV_TRAIN_CPU_THREADS", "4"))
        )
    else:
        # GPU runs still do per-sample CPU tensor math in the dataset path
        # (normalize/stack). Above 96px those ops cross torch's ~32k-element
        # parallel_for grain and EVERY op fans out to all cores — on a
        # 244-core box four workers livelocked at image_size 154 (M3 hang).
        # A modest cap keeps the main-process val path sane; workers get
        # pinned to 1 thread via worker_init below.
        torch.set_num_threads(
            int(os.environ.get("LOCALNAV_TRAIN_CPU_THREADS", "8"))
        )

    from torch.utils.data import ConcatDataset

    from supernav.methods.localnav.dataset import load_samples_jsonl

    data_paths = [config.data] if isinstance(config.data, str) else list(config.data)
    root_paths = [config.root] if isinstance(config.root, str) else list(config.root)
    if len(data_paths) != len(root_paths):
        raise ValueError("--data and --root must be given in matching pairs")

    # Split PER SOURCE (hop-level by default) so every source contributes to
    # val, then concat — sources mix at their natural sample proportions.
    train_sets, val_sets, val_total = [], [], 0
    for data_path, root_path in zip(data_paths, root_paths):
        samples = load_samples_jsonl(data_path)
        if config.use_point_head:
            _attach_point_track(samples, root_path, is_main)
        val_samples, train_samples = split_train_val(
            samples, config.val_fraction, config.seed, config.val_split
        )
        train_samples = oversample_onpolicy_positives(
            train_samples, config.onpolicy_positive_repeat, config.stop_k
        )
        if config.source_weights:
            before = len(train_samples)
            train_samples = oversample_by_root(
                train_samples, root_path, config.source_weights, config.seed
            )
            if is_main and len(train_samples) != before:
                print(
                    f"[localnav-train] source weight on {os.path.basename(str(root_path))}: "
                    f"{before:,} -> {len(train_samples):,} rows",
                    flush=True,
                )
        if config.far_sample_weight > 1.0:
            before = len(train_samples)
            far_before = sum(
                1
                for s in train_samples
                if (s.target_frame_index - s.current_frame_index)
                >= config.far_sample_threshold
            )
            train_samples = oversample_far_targets(
                train_samples,
                config.far_sample_weight,
                config.far_sample_threshold,
                config.seed,
            )
            if is_main:
                far_after = sum(
                    1
                    for s in train_samples
                    if (s.target_frame_index - s.current_frame_index)
                    >= config.far_sample_threshold
                )
                print(
                    f"[localnav-train] far-target weight x{config.far_sample_weight} "
                    f"(>={config.far_sample_threshold} primitives): "
                    f"{before:,} -> {len(train_samples):,} rows, far share "
                    f"{far_before / max(before, 1):.1%} -> "
                    f"{far_after / max(len(train_samples), 1):.1%}",
                    flush=True,
                )
        annotated = annotate_onpolicy_advantages(
            train_samples, config.rl_advantage_beta
        )
        if annotated and is_main:
            print(
                f"[localnav-train] RL-R1 advantage weights on {annotated} "
                f"on-policy samples (beta={config.rl_advantage_beta})",
                flush=True,
            )
        train_sets.append(
            LocalNavTorchDataset(
                train_samples,
                root_path,
                point_jitter_px=config.point_jitter_px,
                sim2real_aug=config.sim2real_aug,
                waypoint_scale=config.waypoint_scale,
                horizon=config.action_horizon,
                zero_motion_stop=config.zero_motion_stop,
                augment=True,
                stop_k=config.stop_k,
                image_size=config.image_size,
                conditioning=config.conditioning,
                context_size=config.context_size,
                context_mode=config.context_mode,
                memory_budget=config.memory_budget,
                memory_recent_tail=config.memory_recent_tail,
                memory_anchor=config.memory_anchor,
                memory_mid_span=config.memory_mid_span,
                memory_mid_count=config.memory_mid_count,
                hindsight_action_weight=config.hindsight_action_weight,
            )
        )
        if val_samples:
            val_total += len(val_samples)
            val_sets.append(
                LocalNavTorchDataset(
                    val_samples, root_path, augment=False,
                    waypoint_scale=config.waypoint_scale, stop_k=config.stop_k,
                    horizon=config.action_horizon,
                    zero_motion_stop=config.zero_motion_stop,
                    image_size=config.image_size,
                    conditioning=config.conditioning,
                    context_size=config.context_size,
                    context_mode=config.context_mode,
                    memory_budget=config.memory_budget,
                    memory_recent_tail=config.memory_recent_tail,
                    memory_anchor=config.memory_anchor,
                    memory_mid_span=config.memory_mid_span,
                    memory_mid_count=config.memory_mid_count,
                    hindsight_action_weight=config.hindsight_action_weight,
                )
            )

    # The GLOBAL batch (and thus the optimization trajectory) is config.batch
    # regardless of world size — DDP splits it across ranks instead of
    # multiplying it, so single-GPU and multi-GPU runs stay comparable.
    per_rank_batch = config.batch
    workers = config.workers
    if distributed:
        if config.batch % world_size:
            raise ValueError(
                f"--batch {config.batch} must be divisible by world size {world_size}"
            )
        per_rank_batch = config.batch // world_size
        workers = max(2, config.workers // world_size)

    train_dataset = (
        train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    )
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.seed,
        )
    def _single_thread_worker(_worker_id: int) -> None:
        torch.set_num_threads(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_rank_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        drop_last=len(train_dataset) >= config.batch,
        pin_memory=device.type == "cuda",
        worker_init_fn=_single_thread_worker if workers > 0 else None,
        collate_fn=_gca_collate if config.context_mode == "gca" else None,
    )
    val_loader = None
    if val_sets and is_main:  # validation is a rank-0 duty
        val_dataset = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
        # Cap the val pass: 5% of a multi-million-sample mix is a >100k-image
        # epoch that the other ranks would sit through at a barrier. A fixed
        # deterministic subsample keeps the metric comparable run-to-run
        if len(val_dataset) > config.max_val_samples > 0:
            keep = torch.randperm(
                len(val_dataset), generator=torch.Generator().manual_seed(config.seed)
            )[: config.max_val_samples].tolist()
            val_dataset = torch.utils.data.Subset(val_dataset, keep)
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(config.batch, max(1, val_total)),
            shuffle=False,
            num_workers=0,
            collate_fn=_gca_collate if config.context_mode == "gca" else None,
        )

    model = NomadPolicy(
        use_coord_embed=config.use_coord_embed,
        use_dist_head=config.use_dist_head,
        use_point_head=config.use_point_head,
        encoder=config.encoder,
        image_size=config.image_size,
        head_type=config.head_type,
        context_size=config.context_size,
        context_mode=config.context_mode,
        memory_budget=config.memory_budget,
        action_horizon=config.action_horizon,
    ).to(device)
    train_model: torch.nn.Module = model
    if distributed:
        from torch.nn.parallel import DistributedDataParallel

        # find_unused_parameters: null_goal_token only joins the graph on
        # batches where condition dropout actually fires (and never with
        # cfg_drop_prob=0) — without this flag DDP errors on those steps.
        train_model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    if str(config.encoder).endswith("-finetune"):
        # Pretrained ViT trunks want a gentler lr than freshly-initialized
        # heads — full lr on a finetuned trunk erases the pretraining.
        trunk_params, rest_params = [], []
        for name, param in model.named_parameters():
            (trunk_params if ".backbone." in name else rest_params).append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": rest_params, "lr": config.lr},
                {"params": trunk_params, "lr": config.lr * 0.1},
            ],
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, config.warmup, config.steps)
    )
    ema = _Ema(model, config.ema_decay)
    num_timesteps = model.scheduler.num_train_timesteps
    use_amp = bool(config.amp) and device.type == "cuda"
    pos_weight = torch.tensor(float(config.stop_pos_weight), device=device)

    out_dir = Path(config.out)
    log_path = out_dir / "train_log.jsonl"
    ckpt_path = out_dir / "ckpt_latest.pt"
    if ckpt_path.exists() and not config.overwrite:
        raise RuntimeError(
            f"refusing to overwrite {ckpt_path} — experiment run dirs are "
            "append-only (every config/ckpt/record is kept); point --out at a "
            "fresh exp<NN> directory, or pass --overwrite to replace this run"
        )
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(
            json.dumps(
                {
                    "train_config": asdict(config),
                    "world_size": world_size,
                    "git_commit": _git_commit(),
                    "data_manifest_sha256": [
                        _manifest_sha256(root) for root in root_paths
                    ],
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
                indent=2,
            )
        )

    def _save(step: int) -> None:
        torch.save(
            {
                "state_dict": ema.state_dict(),
                "raw_state_dict": model.state_dict(),
                "step": step,
                "train_config": asdict(config),
                "waypoint_scale": config.waypoint_scale,
            },
            ckpt_path,
        )

    def _val_metrics() -> "dict | None":
        if val_loader is None:
            return None
        model.eval()
        losses, eps_losses, stop_losses = [], [], []
        probs, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                stop = batch["stop"].to(device)
                cond = model.encode(
                    batch["context"].to(device),
                    batch["goal_pair"].to(device),
                    frame_ages=batch["frame_ages"].to(device)
                    if "frame_ages" in batch
                    else None,
                    pad_mask=batch["pad_mask"].to(device)
                    if "pad_mask" in batch
                    else None,
                    goal_point=batch["goal_point"].to(device)
                    if config.use_coord_embed
                    else None,
                )
                actions = batch["actions"].to(device)
                noise = torch.randn_like(actions)
                if config.head_type == "flow":
                    t_flow = torch.rand(actions.shape[0], device=device)
                    timesteps = t_flow * FLOW_TIME_SCALE
                    noisy = (
                        (1.0 - t_flow.view(-1, 1, 1)) * actions
                        + t_flow.view(-1, 1, 1) * noise
                    )
                    eps_target = noise - actions
                else:
                    timesteps = torch.randint(
                        0, num_timesteps, (actions.shape[0],), device=device
                    )
                    noisy = model.scheduler.add_noise(actions, noise, timesteps)
                    eps_target = noise
                eps = model.predict_noise(noisy, timesteps, cond)
                stop_logit = model.predict_stop_logit(cond)
                eps_loss = torch.nn.functional.mse_loss(eps, eps_target)
                s_loss = stop_bce_loss(
                    stop_logit, stop, torch.ones_like(stop, dtype=torch.bool),
                    pos_weight=pos_weight,
                )
                eps_losses.append(float(eps_loss))
                stop_losses.append(float(s_loss))
                losses.append(float(eps_loss + config.stop_loss_weight * s_loss))
                probs.extend(torch.sigmoid(stop_logit).cpu().numpy().tolist())
                labels.extend(stop.cpu().numpy().tolist())
        model.train()
        precision, recall = stop_precision_recall(
            np.asarray(probs), np.asarray(labels)
        )
        return {
            "val_loss": round(float(np.mean(losses)), 6),
            "val_eps_loss": round(float(np.mean(eps_losses)), 6),
            "val_stop_loss": round(float(np.mean(stop_losses)), 6),
            "val_stop_precision": round(precision, 4),
            "val_stop_recall": round(recall, 4),
        }

    train_model.train()
    iterator = _infinite(train_loader, sampler)
    final_loss = float("nan")
    started = time.time()
    for step in range(1, config.steps + 1):
        batch = next(iterator)
        actions = batch["actions"].to(device)
        stop = batch["stop"].to(device)
        batch_size = actions.shape[0]
        drop_goal = None
        keep_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        if config.cfg_drop_prob > 0.0:
            drop_goal = torch.rand(batch_size, device=device) < config.cfg_drop_prob
            keep_mask = ~drop_goal
        noise = torch.randn_like(actions)
        if config.head_type == "flow":
            # Rectified flow: x_t = (1−t)·x0 + t·ε, net target v = ε − x0.
            # Continuous t rides the same timestep embedding at t*10.
            t_flow = torch.rand(batch_size, device=device)
            timesteps = t_flow * FLOW_TIME_SCALE
            noisy = (1.0 - t_flow.view(-1, 1, 1)) * actions + t_flow.view(-1, 1, 1) * noise
            eps_target = noise - actions
        else:
            timesteps = torch.randint(0, num_timesteps, (batch_size,), device=device)
            noisy = model.scheduler.add_noise(actions, noise, timesteps)
            eps_target = noise

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            # Through train_model.__call__ (NOT model.encode) so DDP's
            # gradient-sync hooks fire — DDP only instruments forward().
            eps_pred, stop_logit, dist_pred, point_pred = train_model(
                batch["context"].to(device),
                batch["goal_pair"].to(device),
                noisy,
                timesteps,
                goal_point=batch["goal_point"].to(device)
                if config.use_coord_embed
                else None,
                drop_goal=drop_goal,
                frame_ages=batch["frame_ages"].to(device)
                if "frame_ages" in batch
                else None,
                pad_mask=batch["pad_mask"].to(device)
                if "pad_mask" in batch
                else None,
            )
            if config.rl_advantage_beta > 0.0:
                weights = batch["weight"].to(device)
                per_sample = torch.nn.functional.mse_loss(
                    eps_pred, eps_target.to(eps_pred.dtype), reduction="none"
                ).mean(dim=(1, 2))
                eps_loss = (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)
            else:
                eps_loss = torch.nn.functional.mse_loss(
                    eps_pred, eps_target.to(eps_pred.dtype)
                )
            s_loss = stop_bce_loss(
                stop_logit.float(), stop, keep_mask, pos_weight=pos_weight
            )
            loss = eps_loss + config.stop_loss_weight * s_loss
            d_loss = None
            if dist_pred is not None and config.dist_loss_weight > 0.0:
                if bool(keep_mask.any()):
                    d_loss = torch.nn.functional.mse_loss(
                        dist_pred[keep_mask].float(),
                        batch["distance"].to(device)[keep_mask],
                    )
                    loss = loss + config.dist_loss_weight * d_loss
            p_loss = None
            if point_pred is not None and config.point_loss_weight > 0.0:
                track = batch["point_track"].to(device)  # (B, 3): u, v, vis
                vis_label = track[:, 2]
                # Mask unavailable labels and rows whose goal conditioning was dropped.
                known = (vis_label >= 0.0) & keep_mask
                if bool(known.any()):
                    p_vis = torch.nn.functional.binary_cross_entropy_with_logits(
                        point_pred[known, 2].float(), vis_label[known]
                    )
                    p_loss = p_vis
                    uv_mask = known & (vis_label > 0.5)
                    if bool(uv_mask.any()):
                        p_uv = torch.nn.functional.smooth_l1_loss(
                            point_pred[uv_mask, :2].float(), track[uv_mask, :2]
                        )
                        p_loss = p_loss + p_uv
                    loss = loss + config.point_loss_weight * p_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(model)
        final_loss = float(loss)

        if is_main and (step % config.log_every == 0 or step == config.steps):
            record = {
                "step": step,
                "loss": round(final_loss, 6),
                "eps_loss": round(float(eps_loss), 6),
                "stop_loss": round(float(s_loss), 6),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.time() - started, 1),
            }
            if d_loss is not None:
                record["dist_loss"] = round(float(d_loss), 6)
            if p_loss is not None:
                record["point_loss"] = round(float(p_loss), 6)
            if step % config.save_every == 0 or step == config.steps:
                val = _val_metrics()
                if val is not None:
                    record.update(val)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"[localnav-train] {record}", flush=True)
        if is_main and (step % config.save_every == 0 or step == config.steps):
            _save(step)

    if distributed:
        import torch.distributed as dist

        dist.barrier()  # keep every rank alive until rank 0 finished saving
        dist.destroy_process_group()
    return {
        "steps": config.steps,
        "final_loss": final_loss,
        "ckpt_path": str(ckpt_path),
        "log_path": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--stop-loss-weight", type=float, default=0.5)
    parser.add_argument("--stop-pos-weight", type=float, default=3.0)
    parser.add_argument("--stop-k", type=int, default=2)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.0)
    parser.add_argument("--use-coord-embed", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-split", choices=("hop", "sample"), default="hop")
    parser.add_argument("--onpolicy-positive-repeat", type=int, default=1)
    parser.add_argument(
        "--source-weight", action="append", default=[], metavar="SUBSTRING=WEIGHT",
        help="repeat a data source so it takes a larger share of the fixed step budget",
    )
    parser.add_argument(
        "--sim2real-aug", type=float, default=0.0,
        help="per-frame exposure/gamma/noise/blur strength for real-camera transfer (0 = off)",
    )
    parser.add_argument(
        "--far-sample-weight", type=float, default=1.0,
        help="sampling mass multiplier for far-target rows (1.0 = uniform, current behavior)",
    )
    parser.add_argument(
        "--far-sample-threshold", type=int, default=16,
        help="remaining primitives that count as far (16 primitives is about 4 m)",
    )
    parser.add_argument("--use-dist-head", action="store_true")
    parser.add_argument("--dist-loss-weight", type=float, default=0.0)
    parser.add_argument("--use-point-head", action="store_true")
    parser.add_argument("--point-loss-weight", type=float, default=0.0)
    parser.add_argument("--rl-advantage-beta", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--point-jitter", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--encoder", default="effnet-scratch",
        choices=(
            "effnet-scratch", "effnet-imagenet", "dino-s-frozen",
            "effnet-b4-imagenet", "convnext-tiny-imagenet",
            "dino-s-finetune", "siglip-b16-finetune",
        ),
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--waypoint-scale", type=float, default=DEFAULT_WAYPOINT_SCALE_M,
                        help="Meters per waypoint unit; sets the planning horizon with --action-horizon")
    parser.add_argument("--zero-motion-stop", action="store_true",
                        help="Mark zero future displacement as reached")
    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON,
                        help="Number of output waypoints; serving reads train_config from the checkpoint")
    parser.add_argument("--head-type", default="ddpm", choices=("ddpm", "flow"))
    parser.add_argument(
        "--conditioning", default="marker", choices=("marker", "crop", "coord"),
    )
    parser.add_argument("--flow-infer-steps", type=int, default=4)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument(
        "--context-mode", default="fixed", choices=("fixed", "gca"),
        help="gca = variable-length memory (anchor + adaptive-stride + "
        "current) with sinusoidal age encoding",
    )
    parser.add_argument("--memory-budget", type=int, default=12)
    parser.add_argument("--memory-recent-tail", type=int, default=0)
    parser.add_argument("--memory-anchor", type=int, default=1, choices=(0, 1))
    # Zero mid_span uses all history; a positive span limits selection to recent steps.
    parser.add_argument("--memory-mid-span", type=int, default=0)
    # Zero mid_count uses stride thinning; a positive count selects evenly spaced frames.
    parser.add_argument("--memory-mid-count", type=int, default=0)
    # Zero hindsight_action_weight excludes hindsight samples from action supervision.
    parser.add_argument("--hindsight-action-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.conditioning == "coord" and not args.use_coord_embed:
        # coord conditioning = the point rides ONLY through the coord token.
        args.use_coord_embed = True
    config = TrainConfig(
        data=args.data[0] if len(args.data) == 1 else tuple(args.data),
        root=args.root[0] if len(args.root) == 1 else tuple(args.root),
        out=args.out,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        warmup=args.warmup,
        ema_decay=args.ema_decay,
        stop_loss_weight=args.stop_loss_weight,
        stop_pos_weight=args.stop_pos_weight,
        stop_k=args.stop_k,
        cfg_drop_prob=args.cfg_drop_prob,
        use_coord_embed=args.use_coord_embed,
        use_dist_head=args.use_dist_head,
        dist_loss_weight=args.dist_loss_weight,
        use_point_head=args.use_point_head,
        point_loss_weight=args.point_loss_weight,
        rl_advantage_beta=args.rl_advantage_beta,
        amp=args.amp,
        overwrite=args.overwrite,
        val_fraction=args.val_fraction,
        val_split=args.val_split,
        onpolicy_positive_repeat=args.onpolicy_positive_repeat,
        far_sample_weight=args.far_sample_weight,
        sim2real_aug=args.sim2real_aug,
        source_weights=tuple(args.source_weight),
        far_sample_threshold=args.far_sample_threshold,
        device=args.device,
        workers=args.workers,
        point_jitter_px=args.point_jitter,
        waypoint_scale=args.waypoint_scale,
        action_horizon=args.action_horizon,
        zero_motion_stop=args.zero_motion_stop,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        encoder=args.encoder,
        image_size=args.image_size,
        head_type=args.head_type,
        conditioning=args.conditioning,
        flow_infer_steps=args.flow_infer_steps,
        context_size=args.context_size,
        context_mode=args.context_mode,
        memory_budget=args.memory_budget,
        memory_recent_tail=args.memory_recent_tail,
        memory_anchor=bool(args.memory_anchor),
        memory_mid_span=args.memory_mid_span,
        memory_mid_count=args.memory_mid_count,
        hindsight_action_weight=args.hindsight_action_weight,
    )
    result = run_training(config)
    print(f"[localnav-train] DONE {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
