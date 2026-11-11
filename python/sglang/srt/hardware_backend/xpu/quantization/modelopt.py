from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.nvfp4_xpu import dequantize_nvfp4
from sglang.srt.layers.parameter import ModelWeightParameter, PerTensorScaleParameter
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4A16LinearMethod,
    ModelOptNvFp4FusedMoEMethod,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config

logger = logging.getLogger(__name__)


def _validate_scales(scales: torch.Tensor) -> None:
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError(
            "ModelOpt NVFP4 global multipliers must be positive and finite"
        )


def _dequantize(
    *,
    packed: torch.Tensor,
    scales: torch.Tensor,
    multiplier: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return dequantize_nvfp4(
        packed=packed, scales=scales, global_multiplier=multiplier, dtype=dtype
    )


class XPUModelOptNvFp4A16LinearMethod(ModelOptNvFp4A16LinearMethod):
    def __init__(self, quant_config: ModelOptFp4Config):
        if quant_config.is_awq:
            raise NotImplementedError(
                "XPU NVFP4 weight-only does not implement AWQ input prescaling"
            )
        super().__init__(quant_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if (
            not self.quant_config.is_checkpoint_nvfp4_serialized
            or self.quant_config.group_size != 16
        ):
            raise NotImplementedError(
                "XPU ModelOpt W4A16 requires serialized group-16 NVFP4"
            )
        _validate_scales(layer.weight_scale_2)
        if layer.weight_scale_2.numel() != len(layer.logical_widths):
            raise ValueError("ModelOpt fused scales do not match logical projections")
        if sum(layer.logical_widths) != layer.weight.shape[0]:
            raise ValueError("ModelOpt fused widths do not match packed weight rows")
        if len(layer.logical_widths) == 1:
            layer.xpu_weight_scale_2 = layer.weight_scale_2[0]
        else:
            layer.xpu_weight_scale_2 = layer.weight_scale_2.repeat_interleave(
                torch.tensor(layer.logical_widths, device=layer.weight.device)
            )
        del layer.input_scale
        logger.warning_once(
            "XPU ModelOpt NVFP4 W4A16: original packed weights and multipliers "
            "stay on device; projections dequantize on XPU, without CUDA Marlin."
        )

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = x.new_empty((*x.shape[:-1], layer.weight.shape[0]))
        # Bound dequantization scratch for quantized vocabulary heads.
        for start in range(0, layer.weight.shape[0], 8192):
            stop = min(start + 8192, layer.weight.shape[0])
            multiplier = layer.xpu_weight_scale_2
            if multiplier.numel() != 1:
                multiplier = multiplier[start:stop]
            weight = _dequantize(
                packed=layer.weight[start:stop],
                scales=layer.weight_scale[start:stop],
                multiplier=multiplier,
                dtype=x.dtype,
            )
            output[..., start:stop] = F.linear(
                x, weight, None if bias is None else bias[start:stop]
            )
        return output


class XPUModelOptNvFp4MoEMethod(ModelOptNvFp4FusedMoEMethod):
    def __init__(self, quant_config: ModelOptFp4Config):
        from sglang.srt.layers.moe import get_moe_runner_backend

        if quant_config.is_awq:
            raise NotImplementedError(
                "XPU NVFP4 weight-only does not implement AWQ input prescaling"
            )
        if (
            not quant_config.is_checkpoint_nvfp4_serialized
            or quant_config.group_size != 16
        ):
            raise NotImplementedError(
                "XPU ModelOpt W4A16 requires serialized group-16 NVFP4"
            )
        backend = get_moe_runner_backend()
        if not (backend.is_auto() or backend.is_triton()):
            raise NotImplementedError(
                "XPU ModelOpt direct experts do not implement CUDA MoE runners"
            )
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if extra_weight_attrs.get("with_bias", False):
            raise NotImplementedError(
                "XPU ModelOpt direct experts do not implement expert bias"
            )
        if hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError("ModelOpt NVFP4 expert dimensions must be divisible by 16")
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.params_dtype = params_dtype
        layer.quant_config = self.quant_config
        loader = extra_weight_attrs["weight_loader"]
        intermediate = intermediate_size_per_partition
        num_shards = 2 if layer.moe_runner_config.is_gated else 1
        for name, shape, dtype in (
            (
                "w13_weight",
                (num_experts, num_shards * intermediate, hidden_size // 2),
                torch.uint8,
            ),
            ("w2_weight", (num_experts, hidden_size, intermediate // 2), torch.uint8),
            (
                "w13_weight_scale",
                (num_experts, num_shards * intermediate, hidden_size // 16),
                torch.float8_e4m3fn,
            ),
            (
                "w2_weight_scale",
                (num_experts, hidden_size, intermediate // 16),
                torch.float8_e4m3fn,
            ),
        ):
            layer.register_parameter(
                name,
                ModelWeightParameter(
                    data=torch.empty(shape, dtype=dtype),
                    input_dim=1,
                    output_dim=2,
                    weight_loader=loader,
                ),
            )
        for name, shape in (
            ("w13_weight_scale_2", (num_experts, num_shards)),
            ("w2_weight_scale_2", (num_experts,)),
            ("w13_input_scale", (num_experts, num_shards)),
            ("w2_input_scale", (num_experts,)),
        ):
            layer.register_parameter(
                name,
                PerTensorScaleParameter(
                    data=torch.empty(shape, dtype=torch.float32),
                    weight_loader=loader,
                ),
            )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        if layer.moe_tp_size != 1 or layer.moe_ep_size != 1:
            raise NotImplementedError(
                "XPU ModelOpt direct experts require TP=1 and EP=1"
            )
        if (
            (moe_runner_config.activation, moe_runner_config.is_gated)
            not in (("silu", True), ("relu2", False))
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
                "XPU ModelOpt direct experts require plain gated SiLU or non-gated ReLU2, "
                "with output routing weights"
            )
        self.moe_runner_config = moe_runner_config
        self.runner = None

    def get_moe_runner_backend(self, layer: torch.nn.Module):
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _validate_scales(layer.w13_weight_scale_2)
        _validate_scales(layer.w2_weight_scale_2)
        del layer.w13_input_scale
        del layer.w2_input_scale
        logger.warning_once(
            "XPU ModelOpt NVFP4 experts use packed resident weights and on-device "
            "active-expert W4A16 computation; not native FP4 hardware acceleration."
        )

    def apply(
        self, layer: torch.nn.Module, dispatch_output: StandardDispatchOutput
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        ids, weights = topk.topk_ids, topk.topk_weights
        if ids.shape != weights.shape or ids.shape[0] != x.shape[0]:
            raise ValueError("ModelOpt NVFP4 routes do not match activation rows")
        result = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        # Only routing IDs cross to the host; model data and forward stay on XPU.
        for expert in ids.unique().tolist():
            if expert == -1:
                continue
            if not 0 <= expert < layer.w13_weight.shape[0]:
                raise ValueError("ModelOpt NVFP4 route exceeds the local expert pool")
            rows, slots = (ids == expert).nonzero(as_tuple=True)
            projected = F.linear(
                x.index_select(0, rows),
                _dequantize(
                    packed=layer.w13_weight[expert],
                    scales=layer.w13_weight_scale[expert],
                    multiplier=layer.w13_weight_scale_2[expert],
                    dtype=x.dtype,
                ),
            )
            if self.moe_runner_config.is_gated:
                gate, up = projected.chunk(2, dim=-1)
                activated = F.silu(gate) * up
            else:
                activated = F.relu(projected).square()
            output = F.linear(
                activated,
                _dequantize(
                    packed=layer.w2_weight[expert],
                    scales=layer.w2_weight_scale[expert],
                    multiplier=layer.w2_weight_scale_2[expert],
                    dtype=x.dtype,
                ),
            )
            result.index_add_(
                0, rows, output.float() * weights[rows, slots, None].float()
            )
        # FusedMoE folds the ModelOpt routed scaling factor into TopK weights.
        return StandardCombineInput(hidden_states=result.to(x.dtype))
