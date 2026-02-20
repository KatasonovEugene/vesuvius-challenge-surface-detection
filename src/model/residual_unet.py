from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_seq(
    value: Union[int, Sequence[int]], spatial_dims: int, *, name: str
) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * spatial_dims
    value = tuple(value)
    if len(value) != spatial_dims:
        raise ValueError(f"{name} must have length {spatial_dims}, got {len(value)}")
    return value


def _ensure_list(value: Union[int, Sequence[int]], n: int, *, name: str) -> List[int]:
    if isinstance(value, int):
        return [value] * n
    value = list(value)
    if len(value) != n:
        raise ValueError(f"{name} must have length {n}, got {len(value)}")
    return value


@dataclass(frozen=True)
class _Ops:
    conv: type
    conv_transpose: type
    norm: type
    upsample_mode: str


def _get_ops(spatial_dims: int, norm: str) -> _Ops:
    if spatial_dims == 3:
        conv = nn.Conv3d
        conv_transpose = nn.ConvTranspose3d
        upsample_mode = "trilinear"
        if norm == "instance":
            norm_op = nn.InstanceNorm3d
        elif norm == "batch":
            norm_op = nn.BatchNorm3d
        else:
            raise ValueError(f"Unsupported norm={norm!r}. Use 'instance' or 'batch'.")
    elif spatial_dims == 2:
        conv = nn.Conv2d
        conv_transpose = nn.ConvTranspose2d
        upsample_mode = "bilinear"
        if norm == "instance":
            norm_op = nn.InstanceNorm2d
        elif norm == "batch":
            norm_op = nn.BatchNorm2d
        else:
            raise ValueError(f"Unsupported norm={norm!r}. Use 'instance' or 'batch'.")
    else:
        raise ValueError(f"Unsupported spatial_dims={spatial_dims}. Use 2 or 3.")
    return _Ops(conv=conv, conv_transpose=conv_transpose, norm=norm_op, upsample_mode=upsample_mode)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        conv_op: type,
        norm_op: type,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        spatial_dims: int,
        *,
        conv_bias: bool,
        act: nn.Module,
    ):
        super().__init__()
        kernel_size_t = _ensure_seq(kernel_size, spatial_dims, name="kernel_size")
        stride_t = _ensure_seq(stride, spatial_dims, name="stride")
        padding = tuple(k // 2 for k in kernel_size_t)

        self.conv = conv_op(
            in_channels,
            out_channels,
            kernel_size=kernel_size_t,
            stride=stride_t,
            padding=padding,
            bias=conv_bias,
        )
        self.norm = norm_op(out_channels, affine=True)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        conv_op: type,
        conv1x1_op: type,
        norm_op: type,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        spatial_dims: int,
        *,
        conv_bias: bool,
        act: nn.Module,
    ):
        super().__init__()
        self.conv1 = ConvNormAct(
            conv_op,
            norm_op,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            spatial_dims=spatial_dims,
            conv_bias=conv_bias,
            act=act,
        )

        kernel_size_t = _ensure_seq(kernel_size, spatial_dims, name="kernel_size")
        padding = tuple(k // 2 for k in kernel_size_t)
        self.conv2 = conv_op(
            out_channels,
            out_channels,
            kernel_size=kernel_size_t,
            stride=_ensure_seq(1, spatial_dims, name="stride"),
            padding=padding,
            bias=conv_bias,
        )
        self.norm2 = norm_op(out_channels, affine=True)

        stride_t = _ensure_seq(stride, spatial_dims, name="stride")
        needs_proj = (in_channels != out_channels) or any(s != 1 for s in stride_t)
        if needs_proj:
            self.proj = conv1x1_op(
                in_channels,
                out_channels,
                kernel_size=_ensure_seq(1, spatial_dims, name="kernel_size"),
                stride=stride_t,
                padding=0,
                bias=conv_bias,
            )
            self.proj_norm = norm_op(out_channels, affine=True)
        else:
            self.proj = None

        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj_norm(self.proj(identity))
        return self.act(out + identity)


class ResidualUNet(nn.Module):
    """Residual U-Net with a ResNet-style residual encoder.

    This is a simplified implementation inspired by the DKFZ dynamic U-Net family:
    - Residual blocks in the encoder ("ResNet encoder")
    - U-Net decoder with skip connections
    - Optional deep supervision: returns stacked logits during training

    Notes for this repo:
    - Inputs are expected as (B, C, D, H, W) for 3D or (B, C, H, W) for 2D.
    """

    def __init__(
        self,
        *,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
        kernel_size: Union[int, Sequence[int]] = 3,
        strides: Sequence[Union[int, Sequence[int]]] = (1, 2, 2, 2, 2, 2),
        n_blocks_per_stage: Union[int, Sequence[int]] = 2,
        n_blocks_per_stage_decoder: Union[int, Sequence[int]] = 1,
        norm: str = "instance",
        act: str = "leaky_relu",
        conv_bias: bool = True,
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.deep_supervision = deep_supervision
        self.deep_supr_num = int(deep_supr_num)

        ops = _get_ops(spatial_dims, norm=norm)
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
        dec_blocks = _ensure_list(n_blocks_per_stage_decoder, n_stages - 1, name="n_blocks_per_stage_decoder")

        self.stem = ConvNormAct(
            ops.conv,
            ops.norm,
            in_channels,
            features_per_stage[0],
            kernel_size=kernel_size,
            stride=1,
            spatial_dims=spatial_dims,
            conv_bias=conv_bias,
            act=act_op,
        )

        self.encoder_stages = nn.ModuleList()
        in_ch = features_per_stage[0]
        for stage_idx in range(n_stages):
            out_ch = features_per_stage[stage_idx]
            stage_stride = strides_list[stage_idx]
            blocks = []
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
                        spatial_dims=spatial_dims,
                        conv_bias=conv_bias,
                        act=act_op,
                    )
                )
                in_ch = out_ch
            self.encoder_stages.append(nn.Sequential(*blocks))

        self.upconvs = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()

        for stage_idx in range(n_stages - 1, 0, -1):
            stride_here = strides_list[stage_idx]
            stride_t = _ensure_seq(stride_here, spatial_dims, name="stride")
            self.upconvs.append(
                ops.conv_transpose(
                    features_per_stage[stage_idx],
                    features_per_stage[stage_idx - 1],
                    kernel_size=stride_t,
                    stride=stride_t,
                    bias=conv_bias,
                )
            )

            blocks = []
            in_decoder_ch = features_per_stage[stage_idx - 1] * 2
            out_decoder_ch = features_per_stage[stage_idx - 1]
            for block_idx in range(dec_blocks[n_stages - 1 - stage_idx]):
                blocks.append(
                    ResidualBlock(
                        ops.conv,
                        ops.conv,
                        ops.norm,
                        in_channels=in_decoder_ch if block_idx == 0 else out_decoder_ch,
                        out_channels=out_decoder_ch,
                        kernel_size=kernel_size,
                        stride=1,
                        spatial_dims=spatial_dims,
                        conv_bias=conv_bias,
                        act=act_op,
                    )
                )
            self.decoder_stages.append(nn.Sequential(*blocks))

        self.seg_head = ops.conv(features_per_stage[0], out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        if self.deep_supervision:
            # One auxiliary head per decoder stage output (deep -> shallow).
            # The final (highest-res) output is produced by seg_head.
            self.aux_heads = nn.ModuleList(
                [
                    ops.conv(features_per_stage[stage_idx - 1], out_channels, kernel_size=1, stride=1, padding=0, bias=True)
                    for stage_idx in range(n_stages - 1, 0, -1)
                ]
            )
        else:
            self.aux_heads = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        skips: List[torch.Tensor] = []
        out = x
        for stage in self.encoder_stages:
            out = stage(out)
            skips.append(out)

        # Decoder: iterate from deepest skip to shallowest (excluding last skip as it is the bottleneck output)
        aux_logits: List[torch.Tensor] = []
        dec_out = skips[-1]
        n_decode = len(self.decoder_stages)
        for i in range(n_decode):
            up = self.upconvs[i](dec_out)
            skip = skips[-2 - i]

            # Handle potential shape mismatches due to odd input sizes
            if up.shape[2:] != skip.shape[2:]:
                up = F.interpolate(up, size=skip.shape[2:], mode=self._upsample_mode, align_corners=False)

            dec_in = torch.cat([up, skip], dim=1)
            dec_out = self.decoder_stages[i](dec_in)

            if self.deep_supervision and self.training and self.aux_heads is not None:
                # Save logits for all decoder stages; we'll pick the ones closest to full-res later.
                aux_logits.append(self.aux_heads[i](dec_out))

        final_logits = self.seg_head(dec_out)

        if not (self.deep_supervision and self.training):
            return final_logits

        # Order heads as: main (full-res) first, then auxiliary heads (nearest to full-res first)
        # Limit number of auxiliary heads used.
        if self.deep_supr_num <= 0:
            stacked = final_logits.unsqueeze(1)
            return stacked

        # Prefer auxiliary outputs closest to full resolution.
        aux_logits = list(reversed(aux_logits))[: self.deep_supr_num]
        upsampled_aux = [
            F.interpolate(a, size=final_logits.shape[2:], mode=self._upsample_mode, align_corners=False)
            if a.shape[2:] != final_logits.shape[2:]
            else a
            for a in aux_logits
        ]
        heads = [final_logits] + upsampled_aux
        return torch.stack(heads, dim=1)


class ResidualUNetDetector(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
        kernel_size: Union[int, Sequence[int]] = 3,
        strides: Sequence[Union[int, Sequence[int]]] = (1, 2, 2, 2, 2, 2),
        n_blocks_per_stage: Union[int, Sequence[int]] = (1, 3, 4, 6, 6, 6),
        n_blocks_per_stage_decoder: Union[int, Sequence[int]] = 1,
        norm: str = "instance",
        act: str = "leaky_relu",
        conv_bias: bool = True,
        deep_supervision: bool = True,
        deep_supr_num: int = 1,
    ):
        super().__init__()
        self.backbone = ResidualUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features_per_stage=features_per_stage,
            kernel_size=kernel_size,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            n_blocks_per_stage_decoder=n_blocks_per_stage_decoder,
            norm=norm,
            act=act,
            conv_bias=conv_bias,
            deep_supervision=deep_supervision,
            deep_supr_num=deep_supr_num,
        )

    def forward(self, volume, **batch):
        volume = volume.unsqueeze(1)
        logits = self.backbone(volume)

        if self.training and logits.ndim == 6:
            return {"logits": logits[:, 0], "outputs": logits}

        return {"logits": logits, "outputs": None}
