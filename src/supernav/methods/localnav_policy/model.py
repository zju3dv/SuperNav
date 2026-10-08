"""NoMaD-style point-conditioned diffusion policy nets.

Architecture (paper-faithful shapes, goal-masking intentionally absent — the
project has no exploration mode):
  - EfficientNet-B0 per-frame observation encoder → 256-d tokens
  - EfficientNet-B0 goal-fusion encoder over (current obs ⊕ marked goal) 6ch
  - optional coordinate token: normalized goal point (x, y) → MLP → 256-d
  - 4-layer / 4-head Transformer over [obs×4, goal(, coord)] tokens → 256-d
  - stop MLP head (binary logit: within stop_k primitives of the target)
  - ConditionalUnet1D diffusion head over 8×(Δx, Δy) normalized action chunks

Classifier-free guidance: goal conditioning (goal + coord tokens) can be
dropped per-row during training (learned null token) and traded against the
conditional prediction at sampling time via ``guidance_scale``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from supernav.methods.localnav_policy.scheduler import SquareCosineNoiseScheduler

CONTEXT_SIZE = 4  # current frame + 3 past frames
ACTION_HORIZON = 8
ACTION_DIM = 2
EMBED_DIM = 256
DEFAULT_WAYPOINT_SCALE_M = 0.25  # meters per unit of normalized waypoint delta
# Flow-matching time t∈[0,1] is fed to the timestep embedding as t*10 so its
# frequency band matches the DDPM integer range the embedding was sized for.
FLOW_TIME_SCALE = 10.0


def _efficientnet_encoder(
    in_channels: int,
    out_dim: int = EMBED_DIM,
    pretrained: bool = False,
    arch: str = "b0",
) -> nn.Module:
    from torchvision.models import efficientnet_b0, efficientnet_b4

    weights = None
    if arch == "b4":
        if pretrained:
            from torchvision.models import EfficientNet_B4_Weights

            weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        net = efficientnet_b4(weights=weights)
        feat_dim = 1792
    else:
        if pretrained:
            from torchvision.models import EfficientNet_B0_Weights

            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        net = efficientnet_b0(weights=weights)
        feat_dim = 1280
    if in_channels != 3:
        stem = net.features[0][0]
        new_stem = nn.Conv2d(
            in_channels,
            stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=stem.stride,
            padding=stem.padding,
            bias=False,
        )
        if pretrained and in_channels % 3 == 0:
            # Tile the pretrained 3ch stem across channel groups (mean-
            # preserving) so ImageNet features survive the 6ch surgery.
            with torch.no_grad():
                repeat = in_channels // 3
                new_stem.weight.copy_(
                    stem.weight.repeat(1, repeat, 1, 1) / float(repeat)
                )
        net.features[0][0] = new_stem
    net.classifier = nn.Identity()
    return nn.Sequential(net, nn.Linear(feat_dim, out_dim))


def _convnext_tiny_encoder(
    in_channels: int, out_dim: int = EMBED_DIM, pretrained: bool = False
) -> nn.Module:
    from torchvision.models import convnext_tiny

    weights = None
    if pretrained:
        from torchvision.models import ConvNeXt_Tiny_Weights

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    net = convnext_tiny(weights=weights)
    if in_channels != 3:
        stem = net.features[0][0]
        new_stem = nn.Conv2d(
            in_channels, stem.out_channels,
            kernel_size=stem.kernel_size, stride=stem.stride,
        )
        if pretrained and in_channels % 3 == 0:
            with torch.no_grad():
                repeat = in_channels // 3
                new_stem.weight.copy_(
                    stem.weight.repeat(1, repeat, 1, 1) / float(repeat)
                )
                new_stem.bias.copy_(stem.bias)
        net.features[0][0] = new_stem
    # classifier = [LayerNorm2d, Flatten, Linear(768,1000)] — swap the Linear
    net.classifier[2] = nn.Linear(net.classifier[2].in_features, out_dim)
    return net


class DinoDualEncoder(nn.Module):
    """DINOv2-S backbone shared by both branches. frozen=True reproduces the
    original judged-negative variant (trunk in eval, no grads); frozen=False
    finetunes the trunk (pair with a reduced trunk lr in the optimizer).
    Inputs must be multiples of the 14px patch size."""

    def __init__(self, out_dim: int = EMBED_DIM, frozen: bool = True) -> None:
        super().__init__()
        self.frozen = bool(frozen)
        self.backbone = _load_dinov2_vits14()
        if self.frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        self.obs_proj = nn.Linear(384, out_dim)
        self.goal_proj = nn.Sequential(
            nn.Linear(768, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

    def train(self, mode: bool = True):  # frozen variant keeps trunk in eval
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def features(self, images: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                return self.backbone(images)
        return self.backbone(images)

    def encode_obs(self, images: torch.Tensor) -> torch.Tensor:
        return self.obs_proj(self.features(images))

    def encode_goal_pair(self, goal_pair: torch.Tensor) -> torch.Tensor:
        current, goal = goal_pair[:, :3], goal_pair[:, 3:]
        return self.goal_proj(
            torch.cat([self.features(current), self.features(goal)], dim=-1)
        )


class SiglipDualEncoder(nn.Module):
    """SigLIP ViT-B/16 trunk (timm, HF-cached) shared by both branches,
    finetuned. Same interface as DinoDualEncoder so it plugs into the same
    slot. Inputs must be multiples of the 16px patch size."""

    def __init__(self, out_dim: int = EMBED_DIM, image_size: int = 160) -> None:
        super().__init__()
        import os

        os.environ.setdefault("HF_HOME", "/vepfs-pykaxon/jingyi/hf_home")
        import timm

        self.backbone = timm.create_model(
            "vit_base_patch16_siglip_224",
            pretrained=True,
            num_classes=0,
            img_size=int(image_size),
        )
        feat = int(getattr(self.backbone, "num_features", 768))
        self.obs_proj = nn.Linear(feat, out_dim)
        self.goal_proj = nn.Sequential(
            nn.Linear(feat * 2, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def encode_obs(self, images: torch.Tensor) -> torch.Tensor:
        return self.obs_proj(self.features(images))

    def encode_goal_pair(self, goal_pair: torch.Tensor) -> torch.Tensor:
        current, goal = goal_pair[:, :3], goal_pair[:, 3:]
        return self.goal_proj(
            torch.cat([self.features(current), self.features(goal)], dim=-1)
        )


def _load_dinov2_vits14():
    import os

    os.environ.setdefault("TORCH_HOME", "/vepfs-pykaxon/jingyi/torch_home")
    try:
        return torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True
        )
    except Exception:  # offline pod: fall back to the vepfs hub cache
        local = os.path.join(
            torch.hub.get_dir(), "facebookresearch_dinov2_main"
        )
        return torch.hub.load(
            local, "dinov2_vits14", source="local", trust_repo=True
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        angles = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class _Conv1dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, n_groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.GroupNorm(min(n_groups, out_ch), out_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _CondResBlock1D(nn.Module):
    """Residual 1D block with FiLM conditioning (scale, bias) from cond vector."""

    def __init__(
        self, in_ch: int, out_ch: int, cond_dim: int, kernel: int, n_groups: int
    ) -> None:
        super().__init__()
        self.block1 = _Conv1dBlock(in_ch, out_ch, kernel, n_groups)
        self.block2 = _Conv1dBlock(out_ch, out_ch, kernel, n_groups)
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_ch * 2))
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        scale, bias = self.cond_encoder(cond).unsqueeze(-1).chunk(2, dim=1)
        out = out * (1.0 + scale) + bias
        out = self.block2(out)
        return out + self.residual(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int = ACTION_DIM,
        global_cond_dim: int = EMBED_DIM,
        diffusion_step_embed_dim: int = 128,
        down_dims: "tuple[int, ...]" = (64, 128, 256),
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        dims = [input_dim, *down_dims]
        pairs = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(pairs):
            is_last = index == len(pairs) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        _CondResBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                        _CondResBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )
        mid_dim = dims[-1]
        self.mid1 = _CondResBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups)
        self.mid2 = _CondResBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups)

        # One upsample per performed downsample (the shallowest skip stays
        # unconsumed — same as the reference ConditionalUnet1D).
        self.ups = nn.ModuleList()
        for dim_in, dim_out in reversed(pairs[1:]):
            self.ups.append(
                nn.ModuleList(
                    [
                        _CondResBlock1D(
                            dim_out * 2, dim_in, cond_dim, kernel_size, n_groups
                        ),
                        _CondResBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                        nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1),
                    ]
                )
            )
        self.final = nn.Sequential(
            _Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor
    ) -> torch.Tensor:
        """sample: [B, T, action_dim]; timestep: [B] or scalar; cond: [B, C]."""
        x = sample.movedim(-1, -2)  # → [B, C, T]
        if timestep.dim() == 0:
            timestep = timestep.expand(x.shape[0])
        cond = torch.cat([self.step_encoder(timestep), global_cond], dim=-1)

        skips = []
        for block1, block2, down in self.downs:
            x = block1(x, cond)
            x = block2(x, cond)
            skips.append(x)
            x = down(x)
        x = self.mid1(x, cond)
        x = self.mid2(x, cond)
        for block1, block2, up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block1(x, cond)
            x = block2(x, cond)
            x = up(x)
        del skips  # shallowest skip intentionally unconsumed (reference parity)
        return self.final(x).movedim(-2, -1)


class NomadPolicy(nn.Module):
    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        context_size: int = CONTEXT_SIZE,
        context_mode: str = "fixed",
        memory_budget: int = 12,
        action_horizon: int = ACTION_HORIZON,
        num_train_timesteps: int = 10,
        use_coord_embed: bool = False,
        use_dist_head: bool = False,
        use_point_head: bool = False,
        encoder: str = "effnet-scratch",
        image_size: int = 96,
        head_type: str = "ddpm",
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.context_mode = str(context_mode)
        if self.context_mode not in ("fixed", "gca"):
            raise ValueError(f"unknown context_mode {context_mode!r}")
        self.memory_budget = int(memory_budget)
        self.action_horizon = action_horizon
        self.use_coord_embed = bool(use_coord_embed)
        self.use_dist_head = bool(use_dist_head)
        self.use_point_head = bool(use_point_head)
        self.encoder_kind = str(encoder)
        self.image_size = int(image_size)
        self.head_type = str(head_type)
        if self.head_type not in ("ddpm", "flow"):
            raise ValueError(f"unknown head_type {head_type!r}")
        if self.encoder_kind in ("dino-s-frozen", "dino-s-finetune"):
            if self.image_size % 14 != 0:
                raise ValueError("dino-s needs image_size % 14 == 0")
            self.dino = DinoDualEncoder(
                embed_dim, frozen=self.encoder_kind == "dino-s-frozen"
            )
            self.obs_encoder = None
            self.goal_encoder = None
        elif self.encoder_kind == "siglip-b16-finetune":
            if self.image_size % 16 != 0:
                raise ValueError("siglip-b16 needs image_size % 16 == 0")
            self.dino = SiglipDualEncoder(embed_dim, image_size=self.image_size)
            self.obs_encoder = None
            self.goal_encoder = None
        elif self.encoder_kind in ("effnet-scratch", "effnet-imagenet"):
            pretrained = self.encoder_kind == "effnet-imagenet"
            self.dino = None
            self.obs_encoder = _efficientnet_encoder(3, embed_dim, pretrained)
            self.goal_encoder = _efficientnet_encoder(6, embed_dim, pretrained)
        elif self.encoder_kind == "effnet-b4-imagenet":
            self.dino = None
            self.obs_encoder = _efficientnet_encoder(3, embed_dim, True, arch="b4")
            self.goal_encoder = _efficientnet_encoder(6, embed_dim, True, arch="b4")
        elif self.encoder_kind == "convnext-tiny-imagenet":
            self.dino = None
            self.obs_encoder = _convnext_tiny_encoder(3, embed_dim, True)
            self.goal_encoder = _convnext_tiny_encoder(6, embed_dim, True)
        else:
            raise ValueError(f"unknown encoder {encoder!r}")
        if self.context_mode == "gca":
            # Variable-length tokens: slot tables break (slot 3 is F5 one step
            # and F9 the next). Each obs token instead carries a sinusoidal
            # embedding of its AGE (current_index - frame_index, in
            # primitives) projected to embed_dim; the goal token keeps a
            # dedicated learned embedding. Ordering/recency live in the age
            # signal (the reference paper's temporal-RoPE lesson).
            self.age_proj = nn.Linear(embed_dim, embed_dim)
            self.goal_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.obs_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.token_pos_emb = None
        else:
            num_tokens = context_size + 1 + (1 if self.use_coord_embed else 0)
            self.token_pos_emb = nn.Parameter(
                torch.zeros(1, num_tokens, embed_dim)
            )
        # Learned stand-in for "no goal given" — CFG's unconditional branch.
        # Initialized with a small random vector so dropout is a real signal
        # from step 0 (an all-zeros token starts indistinguishable from
        # nothing and slows the conditional/unconditional split).
        self.null_goal_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        if self.use_coord_embed:
            self.coord_encoder = nn.Sequential(
                nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, embed_dim)
            )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.stop_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )
        if self.use_dist_head:
            # AUXILIARY regression: multi-task signal
            # for the shared encoder + temporal-distance telemetry. Stopping
            # stays with the stop head — this head never gates termination.
            self.dist_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
            )
        if self.use_point_head:
            # AUXILIARY goal-point tracking (PixelNav-style): predict where the
            # goal point sits in the CURRENT frame (u, v normalized) plus an
            # in-view logit. Pure multi-task signal for the shared encoder —
            # never consulted at inference; enabled by use_point_head.
            self.point_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 3)
            )
        self.noise_pred = ConditionalUnet1D(
            input_dim=ACTION_DIM, global_cond_dim=embed_dim
        )
        self.scheduler = SquareCosineNoiseScheduler(num_train_timesteps)

    def encode(
        self,
        context: torch.Tensor,
        goal_pair: torch.Tensor,
        *,
        goal_point: "torch.Tensor | None" = None,
        drop_goal: "torch.Tensor | None" = None,
        frame_ages: "torch.Tensor | None" = None,
        pad_mask: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """context: [B, K, 3, H, W]; goal_pair: [B, 6, H, W] → [B, embed_dim].

        goal_point: [B, 2] normalized (x, y) — required iff use_coord_embed.
        drop_goal: [B] bool — rows with True get the learned null token in
        place of ALL goal conditioning (goal-fusion token and coord token).
        """
        batch, k = context.shape[:2]
        flat = context.flatten(0, 1)
        if self.context_mode == "gca" and pad_mask is not None:
            # Pack-encode only the valid frames: padding frames through a
            # BatchNorm encoder contaminate the BATCH statistics of other
            # samples (measured), besides wasting compute.
            valid = (~pad_mask).reshape(-1)
            enc = self.dino.encode_obs if self.dino is not None else self.obs_encoder
            feats = enc(flat[valid])
            obs_flat = feats.new_zeros(flat.shape[0], feats.shape[-1])
            obs_flat[valid] = feats
            obs_tokens = obs_flat.view(batch, k, -1)
        elif self.dino is not None:
            obs_tokens = self.dino.encode_obs(flat).view(batch, k, -1)
        else:
            obs_tokens = self.obs_encoder(flat).view(batch, k, -1)
        if self.dino is not None:
            goal_token = self.dino.encode_goal_pair(goal_pair).unsqueeze(1)
        else:
            goal_token = self.goal_encoder(goal_pair).unsqueeze(1)
        keep = None
        if drop_goal is not None:
            keep = (~drop_goal.view(batch, 1, 1)).to(goal_token.dtype)
            null = self.null_goal_token.expand(batch, -1, -1)
            goal_token = goal_token * keep + null * (1.0 - keep)
        tokens = [obs_tokens, goal_token]
        if self.use_coord_embed:
            if goal_point is None:
                raise ValueError(
                    "use_coord_embed model requires goal_point [B, 2] in encode()"
                )
            coord_token = self.coord_encoder(
                goal_point.to(goal_token.dtype)
            ).unsqueeze(1)
            if keep is not None:
                coord_token = coord_token * keep
            tokens.append(coord_token)
        if self.context_mode == "gca":
            if frame_ages is None:
                raise ValueError("gca encode() requires frame_ages")
            return self.fuse_gca(
                obs_tokens, goal_token, frame_ages, pad_mask=pad_mask
            )
        stacked = torch.cat(tokens, dim=1) + self.token_pos_emb
        return self.transformer(stacked).mean(dim=1)

    def fuse_gca(
        self,
        obs_tokens: torch.Tensor,
        goal_token: torch.Tensor,
        frame_ages: torch.Tensor,
        *,
        pad_mask: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """Variable-length fusion from PRE-ENCODED tokens (single source of
        truth for training encode() and the serving-side token cache).

        obs_tokens: [B, K, D]; goal_token: [B, 1, D] (null-token substitution
        for CFG happens BEFORE this call); frame_ages: [B, K] in primitive
        units; pad_mask: [B, K] bool, True = padding."""
        half = obs_tokens.shape[-1] // 2
        freqs = torch.exp(
            torch.arange(half, device=obs_tokens.device, dtype=torch.float32)
            * (-math.log(200.0) / max(1, half - 1))
        )
        ang = frame_ages.to(torch.float32).unsqueeze(-1) * freqs
        age_emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        obs_tokens = obs_tokens + self.age_proj(age_emb) + self.obs_type_emb
        goal_token = goal_token + self.goal_type_emb
        stacked = torch.cat([obs_tokens, goal_token], dim=1)
        key_pad = None
        if pad_mask is not None:
            goal_pad = torch.zeros(
                (pad_mask.shape[0], 1), dtype=torch.bool, device=pad_mask.device
            )
            key_pad = torch.cat([pad_mask, goal_pad], dim=1)
        out = self.transformer(stacked, src_key_padding_mask=key_pad)
        if key_pad is None:
            return out.mean(dim=1)
        keep = (~key_pad).unsqueeze(-1).to(out.dtype)
        return (out * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)

    def predict_stop_logit(self, cond: torch.Tensor) -> torch.Tensor:
        """Binary stop logit; sigmoid of this is the served done_probability."""
        return self.stop_head(cond).squeeze(-1)

    def predict_distance(self, cond: torch.Tensor) -> torch.Tensor:
        """Auxiliary remaining-primitives regression (use_dist_head only)."""
        return self.dist_head(cond).squeeze(-1)

    def predict_point_track(self, cond: torch.Tensor) -> torch.Tensor:
        """Auxiliary (u, v, vis_logit) of the goal point in the current frame."""
        return self.point_head(cond)

    def forward(
        self,
        context: torch.Tensor,
        goal_pair: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        goal_point: "torch.Tensor | None" = None,
        drop_goal: "torch.Tensor | None" = None,
        frame_ages: "torch.Tensor | None" = None,
        pad_mask: "torch.Tensor | None" = None,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]":
        """One training step's forward:
        (eps_pred, stop_logit, dist_pred|None, point_pred|None).

        The training loop MUST call the model (not encode/predict_* directly)
        so DistributedDataParallel's gradient hooks fire — DDP only wraps
        forward(). Inference keeps using encode()/sample_actions().
        """
        cond = self.encode(
            context, goal_pair, goal_point=goal_point, drop_goal=drop_goal,
            frame_ages=frame_ages, pad_mask=pad_mask,
        )
        dist_pred = self.predict_distance(cond) if self.use_dist_head else None
        point_pred = self.predict_point_track(cond) if self.use_point_head else None
        return (
            self.predict_noise(noisy_actions, timesteps, cond),
            self.predict_stop_logit(cond),
            dist_pred,
            point_pred,
        )

    def predict_noise(
        self, noisy_actions: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        return self.noise_pred(noisy_actions, timestep, cond)

    @torch.no_grad()
    def sample_actions(
        self,
        cond: torch.Tensor,
        num_samples: int = 16,
        denoise_steps: int = 10,
        waypoint_scale: float = DEFAULT_WAYPOINT_SCALE_M,
        generator: "torch.Generator | None" = None,
        uncond: "torch.Tensor | None" = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """cond: [1, embed_dim] → [num_samples, action_horizon, 2] in meters.

        With ``uncond`` (the drop_goal encoding of the same inputs) and
        ``guidance_scale`` ≠ 1, each denoise step runs the conditional and
        unconditional branches in one doubled batch and extrapolates:
        ε = ε_u + s·(ε_c − ε_u). Only meaningful for checkpoints trained
        with condition dropout.

        The DDPM schedule is fixed at ``num_train_timesteps``; ``denoise_steps``
        larger than that is clamped (ancestral sampling cannot skip steps —
        DDIM-style striding lands together with training in M3/M5).
        """
        if cond.shape[0] != 1:
            raise ValueError("sample_actions expects a single context vector")
        device = cond.device
        cond_batch = cond.expand(num_samples, -1)
        uncond_batch = None
        if uncond is not None and guidance_scale != 1.0:
            uncond_batch = uncond.expand(num_samples, -1)
        actions = torch.randn(
            (num_samples, self.action_horizon, ACTION_DIM),
            device=device,
            generator=generator,
        )

        def _predict(x, timestep_value: float):
            timestep = torch.tensor(timestep_value, device=device)
            if uncond_batch is None:
                return self.predict_noise(x, timestep, cond_batch)
            both = self.predict_noise(
                torch.cat([x, x], dim=0),
                timestep,
                torch.cat([cond_batch, uncond_batch], dim=0),
            )
            out_c, out_u = both.chunk(2, dim=0)
            return out_u + guidance_scale * (out_c - out_u)

        if self.head_type == "flow":
            # Rectified flow: x_t = (1-t)x0 + t*eps, net predicts v = eps - x0;
            # Euler-integrate t: 1 → 0. denoise_steps IS meaningful here.
            steps = max(1, int(denoise_steps))
            ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
            for index in range(steps):
                t = float(ts[index])
                delta = float(ts[index] - ts[index + 1])
                velocity = _predict(actions, t * FLOW_TIME_SCALE)
                actions = actions - delta * velocity
            return actions * waypoint_scale

        del denoise_steps  # ancestral DDPM cannot skip steps; see docstring
        for t in self.scheduler.timesteps.tolist():
            eps = _predict(actions, float(t))
            actions = self.scheduler.step(eps, t, actions, generator=generator)
        return actions * waypoint_scale
