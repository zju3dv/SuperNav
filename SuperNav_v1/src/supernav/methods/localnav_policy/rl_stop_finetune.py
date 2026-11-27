#!/usr/bin/env python
"""Decision-theoretic fine-tuning of the stop head."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from supernav.methods.localnav_policy.model import NomadPolicy  # noqa: E402
from supernav.methods.localnav_policy.train import (  # noqa: E402
    _git_commit,
    split_train_val,
    stop_precision_recall,
)
from supernav.methods.localnav_policy.train_dataset import (  # noqa: E402
    LocalNavTorchDataset,
)


def stop_go_rewards(geo: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor]":
    """(R_stop, R_go) per state from privileged remaining distance (meters)."""
    r_stop = torch.where(
        geo <= 1.0,
        torch.ones_like(geo),
        torch.where(geo <= 1.5, torch.zeros_like(geo), -torch.ones_like(geo)),
    )
    r_go = torch.where(
        geo <= 1.0, torch.full_like(geo, -0.3), torch.full_like(geo, 0.05)
    )
    return r_stop, r_go


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--data", nargs="+", required=True, help="on-policy roots")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out)
    ckpt_out = out_dir / "ckpt_latest.pt"
    if ckpt_out.exists():
        raise RuntimeError(f"{ckpt_out} exists — run dirs are append-only")
    out_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.base, map_location="cpu")
    train_config = dict(state.get("train_config") or {})
    image_size = int(train_config.get("image_size", 96))
    conditioning = str(train_config.get("conditioning", "marker"))
    policy = NomadPolicy(
        use_coord_embed=bool(train_config.get("use_coord_embed", False)),
        use_dist_head=bool(train_config.get("use_dist_head", False)),
        encoder=str(train_config.get("encoder", "effnet-scratch")),
        image_size=image_size,
        head_type=str(train_config.get("head_type", "ddpm")),
    )
    policy.load_state_dict(state["state_dict"], strict=True)
    policy = policy.to(device)
    for name, param in policy.named_parameters():
        param.requires_grad = name.startswith("stop_head.")
    policy.train()

    from supernav.methods.localnav.dataset import load_samples_jsonl

    train_sets, val_sets = [], []
    for root in args.data:
        samples = [
            s
            for s in load_samples_jsonl(Path(root) / "samples.jsonl")
            if (s.metadata or {}).get("true_remaining_m") is not None
        ]
        val, train = split_train_val(samples, 0.05, args.seed, "hop")
        train_sets.append(LocalNavTorchDataset(
            train, root, augment=True,
            image_size=image_size, conditioning=conditioning,
        ))
        val_sets.append(LocalNavTorchDataset(
            val, root, augment=False,
            image_size=image_size, conditioning=conditioning,
        ))
        for dataset in (train_sets[-1], val_sets[-1]):
            dataset._geos = [
                float(s.metadata["true_remaining_m"]) for s in dataset._samples
            ]

    class _WithGeo(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, index):
            item = self.base[index]
            item["geo"] = torch.tensor(self.base._geos[index], dtype=torch.float32)
            return item

    train_data = torch.utils.data.ConcatDataset([_WithGeo(d) for d in train_sets])
    val_data = torch.utils.data.ConcatDataset([_WithGeo(d) for d in val_sets])
    loader = DataLoader(
        train_data, batch_size=args.batch, shuffle=True, num_workers=8,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=args.lr
    )

    log_path = out_dir / "train_log.jsonl"
    started = time.time()
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            geo = batch["geo"].to(device)
            with torch.no_grad():
                cond = policy.encode(
                    batch["context"].to(device),
                    batch["goal_pair"].to(device),
                    goal_point=batch["goal_point"].to(device)
                    if policy.use_coord_embed
                    else None,
                )
            pi = torch.sigmoid(policy.predict_stop_logit(cond.detach()))
            r_stop, r_go = stop_go_rewards(geo)
            reward = pi * r_stop + (1.0 - pi) * r_go
            loss = -reward.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 25 == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "J": round(float(reward.mean()), 4),
                    "elapsed_s": round(time.time() - started, 1),
                }
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(f"[rl-stop] {record}", flush=True)

    # Held-out readout at the bandit's native operating point (0.5).
    policy.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in DataLoader(val_data, batch_size=args.batch, num_workers=4):
            cond = policy.encode(
                batch["context"].to(device),
                batch["goal_pair"].to(device),
                goal_point=batch["goal_point"].to(device)
                if policy.use_coord_embed
                else None,
            )
            probs.extend(torch.sigmoid(policy.predict_stop_logit(cond)).cpu().tolist())
            labels.extend((batch["geo"] <= 1.0).float().tolist())
    precision, recall = stop_precision_recall(np.asarray(probs), np.asarray(labels))

    # Write the retrained stop head into BOTH slots of a contract-shaped ckpt.
    new_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    for slot in ("state_dict", "raw_state_dict"):
        merged = dict(state[slot])
        for key, value in new_state.items():
            if key.startswith("stop_head."):
                merged[key] = value
        state[slot] = merged
    state["train_config"] = {
        **train_config,
        "rl_stop_finetune": {
            "base": args.base, "epochs": args.epochs, "lr": args.lr,
            "objective": "full-feedback bandit J",
        },
    }
    torch.save(state, ckpt_out)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "stage": "RL-R2 stop-head bandit",
                "base_ckpt": args.base,
                "data": args.data,
                "val_stop_precision@0.5": round(precision, 4),
                "val_stop_recall@0.5": round(recall, 4),
                "git_commit": _git_commit(),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
    )
    print(
        f"[rl-stop] DONE val P@0.5={precision:.3f} R@0.5={recall:.3f} → {ckpt_out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
