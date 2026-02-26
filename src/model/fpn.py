from __future__ import annotations

from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.residual_unet import (
    ConvNormAct,
    ResidualBlock,
    _ensure_list,
    _ensure_seq,
    _get_ops,
)


class ResidualFPN(nn.Module):
    """Feature Pyramid Network with a residual encoder.

    This model follows the standard FPN pattern:
    - Bottom-up residual encoder producing multi-scale feature maps C_i
    - Top-down pathway with lateral 1x1 projections producing P_i
    - Final segmentation head fuses all pyramid levels to full resolution

    For this repo, inputs are expected as (B, C, D, H, W) for 3D or (B, C, H, W) for 2D.
    """

    def __init__(
        self,
        *,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
        strides: Sequence[Union[int, Sequence[int]]] = (1, 2, 2, 2, 2, 2),
        n_blocks_per_stage: Union[int, Sequence[int]] = (1, 3, 4, 6, 6, 6),
        kernel_size: Union[int, Sequence[int]] = 3,
        fpn_channels: int = 96,
        norm: str = "instance",
        act: str = "leaky_relu",
        conv_bias: bool = True,
        fuse_conv_kernel_size: Union[int, Sequence[int]] = 3,
    ):
        super().__init__()

        self.spatial_dims = int(spatial_dims)
        ops = _get_ops(self.spatial_dims, norm=norm)
        self._upsample_mode = ops.upsample_mode

        if act == "leaky_relu":
            act_op = nn.LeakyReLU(inplace=True)
        elif act == "relu":
            act_op = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported act={act!r}. Use 'leaky_relu' or 'relu'.")

        features_per_stage = list(features_per_stage)
        n_stages = len(features_per_stage)
        if n_stages < 2:
            raise ValueError("features_per_stage must have at least 2 stages")

        strides_list = list(strides)
        if len(strides_list) != n_stages:
            raise ValueError(
                f"strides must have length {n_stages} (one per stage), got {len(strides_list)}"
            )

        enc_blocks = _ensure_list(n_blocks_per_stage, n_stages, name="n_blocks_per_stage")

        self.stem = ConvNormAct(
            ops.conv,
            ops.norm,
            in_channels,
            features_per_stage[0],
            kernel_size=kernel_size,
            stride=1,
            spatial_dims=self.spatial_dims,
            conv_bias=conv_bias,
            act=act_op,
        )

        self.encoder_stages = nn.ModuleList()
        in_ch = features_per_stage[0]
        for stage_idx in range(n_stages):
            out_ch = features_per_stage[stage_idx]
            stage_stride = strides_list[stage_idx]
            blocks: List[nn.Module] = []
            for block_idx in range(enc_blocks[stage_idx]):
                stride_for_block = stage_stride if (stage_idx > 0 and block_idx == 0) else 1
                blocks.append(
                    ResidualBlock(
                        ops.conv,
                        ops.conv,
                        ops.norm,
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel_size,
                        stride=stride_for_block,
                        spatial_dims=self.spatial_dims,
                        conv_bias=conv_bias,
                        act=act_op,
                    )
                )
                in_ch = out_ch
            self.encoder_stages.append(nn.Sequential(*blocks))

        fpn_channels = int(fpn_channels)
        if fpn_channels <= 0:
            raise ValueError("fpn_channels must be > 0")

        self.lateral_convs = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()
        for stage_idx in range(n_stages):
            self.lateral_convs.append(
                ops.conv(
                    features_per_stage[stage_idx],
                    fpn_channels,
                    kernel_size=_ensure_seq(1, self.spatial_dims, name="kernel_size"),
                    stride=_ensure_seq(1, self.spatial_dims, name="stride"),
                    padding=0,
                    bias=True,
                )
            )

            smooth_k = _ensure_seq(3, self.spatial_dims, name="kernel_size")
            self.smooth_convs.append(
                ops.conv(
                    fpn_channels,
                    fpn_channels,
                    kernel_size=smooth_k,
                    stride=_ensure_seq(1, self.spatial_dims, name="stride"),
                    padding=tuple(k // 2 for k in smooth_k),
                    bias=conv_bias,
                )
            )

        fuse_k = _ensure_seq(fuse_conv_kernel_size, self.spatial_dims, name="fuse_conv_kernel_size")
        self.fuse = nn.Sequential(
            ops.conv(
                fpn_channels * n_stages,
                fpn_channels,
                kernel_size=fuse_k,
                stride=_ensure_seq(1, self.spatial_dims, name="stride"),
                padding=tuple(k // 2 for k in fuse_k),
                bias=conv_bias,
            ),
            ops.norm(fpn_channels, affine=True),
            act_op,
        )
        self.seg_head = ops.conv(
            fpn_channels,
            out_channels,
            kernel_size=_ensure_seq(1, self.spatial_dims, name="kernel_size"),
            stride=_ensure_seq(1, self.spatial_dims, name="stride"),
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        c_feats: List[torch.Tensor] = []
        out = x
        for stage in self.encoder_stages:
            out = stage(out)
            c_feats.append(out)

        # Top-down FPN: build p_feats from deep to shallow
        p_feats: List[torch.Tensor] = [None] * len(c_feats)  # type: ignore[assignment]

        last_idx = len(c_feats) - 1
        p = self.lateral_convs[last_idx](c_feats[last_idx])
        p = self.smooth_convs[last_idx](p)
        p_feats[last_idx] = p

        for i in range(last_idx - 1, -1, -1):
            lateral = self.lateral_convs[i](c_feats[i])
            p_up = F.interpolate(p, size=lateral.shape[2:], mode=self._upsample_mode, align_corners=False)
            p = lateral + p_up
            p = self.smooth_convs[i](p)
            p_feats[i] = p

        # Fuse all pyramid levels at full resolution (stage 0)
        target_size = p_feats[0].shape[2:]
        p_upsampled = [
            (pf if pf.shape[2:] == target_size else F.interpolate(pf, size=target_size, mode=self._upsample_mode, align_corners=False))
            for pf in p_feats
        ]
        fused = torch.cat(p_upsampled, dim=1)
        fused = self.fuse(fused)
        return self.seg_head(fused)


class FPNDetector(nn.Module):
    """Repo-compatible detector wrapper for `ResidualFPN`."""

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
        strides: Sequence[Union[int, Sequence[int]]] = (1, 2, 2, 2, 2, 2),
        n_blocks_per_stage: Union[int, Sequence[int]] = (1, 3, 4, 6, 6, 6),
        kernel_size: Union[int, Sequence[int]] = 3,
        fpn_channels: int = 96,
        norm: str = "instance",
        act: str = "leaky_relu",
        conv_bias: bool = True,
        fuse_conv_kernel_size: Union[int, Sequence[int]] = 3,
    ):
        super().__init__()
        self.backbone = ResidualFPN(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features_per_stage=features_per_stage,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            kernel_size=kernel_size,
            fpn_channels=fpn_channels,
            norm=norm,
            act=act,
            conv_bias=conv_bias,
            fuse_conv_kernel_size=fuse_conv_kernel_size,
        )

    def forward(self, volume, **batch):
        volume = volume.unsqueeze(1)
        logits = self.backbone(volume)
        return {"logits": logits, "outputs": None}
