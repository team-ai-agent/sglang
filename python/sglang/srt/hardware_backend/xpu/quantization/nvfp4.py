from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4 import (
    CompressedTensorsW4A16Fp4,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4_moe import (
    CompressedTensorsW4A16Nvfp4MoE,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


def dequantize_nvfp4(
    *,
    packed: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError("NVFP4 requires a two-dimensional uint8 packed matrix")
    rows, half_columns = packed.shape
    columns = half_columns * 2
    if columns % 16 or scales.shape != (rows, columns // 16):
        raise ValueError("NVFP4 requires one FP8 scale per16 input values")
    values = unpack_fp4_from_uint8(packed, rows, columns, dtype=torch.float32)
    values *= scales.float().repeat_interleave(16, dim=-1)
    # compressed-tensors stores the global scale as a divisor.
    values /= global_scale.float()
    return values.to(dtype)


def _validate_global_scales(scales: torch.Tensor) -> None:
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("NVFP4 global scales must be positive and finite")


class XPUCompressedTensorsW4A16Fp4(CompressedTensorsW4A16Fp4):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        scales = layer.weight_global_scale.data
        _validate_global_scales(scales)
        if torch.unique(scales).numel() != 1:
            raise ValueError(
                "NVFP4 fused dense projections require one shared global scale"
            )
        if self.has_input_global_scale:
            del layer.input_global_scale
        logger.warning_once(
            "XPU NVFP4 weight-only execution: packed weights remain on XPU; "
            "active projections are dequantized on device. No CUDA Marlin."
        )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        weight = dequantize_nvfp4(
            packed=layer.weight_packed,
            scales=layer.weight_scale,
            global_scale=layer.weight_global_scale[0],
            dtype=x.dtype,
        )
        return F.linear(x, weight, bias)


class XPUCompressedTensorsW4A16Nvfp4MoE(CompressedTensorsW4A16Nvfp4MoE):
    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        if layer.moe_tp_size != 1 or layer.moe_ep_size != 1:
            raise NotImplementedError(
                "XPU NVFP4 direct experts currently require TP=1 and EP=1"
            )
        if (
            moe_runner_config.activation != "silu"
            or not moe_runner_config.is_gated
            or moe_runner_config.apply_router_weight_on_input
            or moe_runner_config.no_combine
            or moe_runner_config.num_fused_shared_experts
            or any(
                value is not None
                for value in (
                    moe_runner_config.gemm1_alpha,
                    moe_runner_config.gemm1_beta,
                    moe_runner_config.gemm1_clamp_limit,
                    moe_runner_config.swiglu_limit,
                )
            )
        ):
            raise NotImplementedError(
                "XPU NVFP4 direct experts require plain gated SiLU and output routing weights"
            )
        self.moe_runner_config = moe_runner_config
        self.runner = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        gate_up = layer.w13_weight_global_scale.data
        _validate_global_scales(gate_up)
        _validate_global_scales(layer.w2_weight_global_scale.data)
        if not (gate_up == gate_up[:, :1]).all():
            raise ValueError(
                "NVFP4 gate/up projections require shared global scales per expert"
            )
        if self.has_input_global_scale:
            del layer.w13_input_global_scale
            del layer.w2_input_global_scale
        logger.warning_once(
            "XPU NVFP4 MoE uses on-device active-expert dequantization with "
            "packed resident weights; performance is not a validated target."
        )

    def apply_weights(
        self, layer: torch.nn.Module, dispatch_output: StandardDispatchOutput
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        ids, weights = topk.topk_ids, topk.topk_weights
        if ids.shape != weights.shape or ids.shape[0] != x.shape[0]:
            raise ValueError("NVFP4 expert routes do not match activation rows")
        result = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        # Only routing metadata crosses to the host; model data stays on XPU.
        for expert in ids.unique().tolist():
            if expert == -1:
                continue
            if not 0 <= expert < layer.w13_weight_packed.shape[0]:
                raise ValueError("NVFP4 expert route exceeds the local expert pool")
            rows, slots = (ids == expert).nonzero(as_tuple=True)
            gate_up = F.linear(
                x.index_select(0, rows),
                dequantize_nvfp4(
                    packed=layer.w13_weight_packed[expert],
                    scales=layer.w13_weight_scale[expert],
                    global_scale=layer.w13_weight_global_scale[expert, 0],
                    dtype=x.dtype,
                ),
            )
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            output = F.linear(
                hidden,
                dequantize_nvfp4(
                    packed=layer.w2_weight_packed[expert],
                    scales=layer.w2_weight_scale[expert],
                    global_scale=layer.w2_weight_global_scale[expert],
                    dtype=x.dtype,
                ),
            )
            result.index_add_(
                0, rows, output.float() * weights[rows, slots, None].float()
            )
        if self.moe_runner_config.routed_scaling_factor is not None:
            result *= self.moe_runner_config.routed_scaling_factor
        return StandardCombineInput(hidden_states=result.to(x.dtype))
