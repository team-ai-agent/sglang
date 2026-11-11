import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.xpu.quantization.modelopt import (
    XPUModelOptNvFp4A16LinearMethod,
    XPUModelOptNvFp4MoEMethod,
)
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")


def fixture(rows, columns, shift):
    codes = (torch.arange(rows * columns).reshape(rows, columns) + shift) % 16
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    scales = (
        torch.linspace(0.125, 1, rows * (columns // 16))
        .reshape(rows, columns // 16)
        .to(torch.float8_e4m3fn)
    )
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed, scales, table[codes] * scales.float().repeat_interleave(16, dim=-1)


def config():
    return ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        use_per_token_activation=False,
    )


@unittest.skipUnless(torch.xpu.is_available(), "requires Intel XPU")
class TestModelOptNvfp4XPU(CustomTestCase):
    def test_tiled_dense_keeps_distinct_fused_multipliers_and_bias(self):
        """Tiling must preserve unequal fused widths and multiplicative global scales."""
        method = XPUModelOptNvFp4A16LinearMethod(config())
        layer = torch.nn.Module()
        method.create_weights(
            layer=layer,
            input_size_per_partition=16,
            output_partition_sizes=[8200, 8],
            input_size=16,
            output_size=8208,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )
        packed, scales, reference = fixture(8208, 16, 0)
        layer.weight.data.copy_(packed)
        layer.weight_scale.data.copy_(scales)
        layer.weight_scale_2.data.copy_(torch.tensor([0.25, 0.75]))
        layer.to("xpu")
        method.process_weights_after_loading(layer)
        reference[:8200] *= 0.25
        reference[8200:] *= 0.75
        x = torch.linspace(-1, 1, 48).reshape(3, 16).to(torch.bfloat16)
        bias = torch.linspace(-0.1, 0.1, 8208).to(torch.bfloat16)
        expected = F.linear(x, reference.to(torch.bfloat16), bias)
        actual = method.apply(layer, x.to("xpu"), bias.to("xpu"))
        torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-3
        )
        self.assertEqual(layer.weight.dtype, torch.uint8)
        self.assertEqual(layer.weight.shape, packed.shape)

    def test_experts_preserve_gate_up_scale_difference_and_dropped_routes(self):
        """Global multipliers must not collapse to max or become CT-style divisors."""
        method = XPUModelOptNvFp4MoEMethod(config())
        layer = torch.nn.Module()
        layer.moe_tp_size = layer.moe_ep_size = 1
        layer.moe_runner_config = MoeRunnerConfig(routed_scaling_factor=0.75)
        method.create_weights(
            layer=layer,
            num_experts=3,
            hidden_size=32,
            intermediate_size_per_partition=16,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )
        up_ref, down_ref = [], []
        for expert in range(3):
            packed, scales, reference = fixture(32, 32, expert)
            layer.w13_weight.data[expert].copy_(packed)
            layer.w13_weight_scale.data[expert].copy_(scales)
            multipliers = torch.tensor([0.125 * (expert + 1), 0.25 * (expert + 1)])
            layer.w13_weight_scale_2.data[expert].copy_(multipliers)
            reference[:16] *= multipliers[0]
            reference[16:] *= multipliers[1]
            up_ref.append(reference.to(torch.bfloat16))
            packed, scales, reference = fixture(32, 16, expert + 3)
            layer.w2_weight.data[expert].copy_(packed)
            layer.w2_weight_scale.data[expert].copy_(scales)
            layer.w2_weight_scale_2.data[expert] = 0.125 * (expert + 2)
            down_ref.append((reference * (0.125 * (expert + 2))).to(torch.bfloat16))
        layer.to("xpu")
        method.create_moe_runner(layer, layer.moe_runner_config)
        method.process_weights_after_loading(layer)
        x = torch.linspace(-0.1, 0.1, 96).reshape(3, 32).to(torch.bfloat16)
        ids = torch.tensor([[0, 2], [1, 0], [2, -1]])
        weights = torch.tensor([[0.3, 0.7], [0.25, 0.75], [1.0, 0.0]])
        expected = torch.zeros(3, 32)
        for row in range(3):
            for slot in range(2):
                expert = ids[row, slot].item()
                if expert < 0:
                    continue
                gate, up = F.linear(x[row], up_ref[expert]).chunk(2)
                expected[row] += (
                    F.linear(F.silu(gate) * up, down_ref[expert]).float()
                    * weights[row, slot]
                    * 0.75
                )
        dispatch = StandardDispatchOutput(
            hidden_states=x.to("xpu"),
            hidden_states_scale=None,
            topk_output=SimpleNamespace(
                topk_ids=ids.to("xpu"), topk_weights=(weights * 0.75).to("xpu")
            ),
        )
        actual = method.apply(layer, dispatch).hidden_states
        torch.testing.assert_close(
            actual.cpu().float(),
            expected.to(torch.bfloat16).float(),
            rtol=2e-2,
            atol=2e-3,
        )
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)

    def test_relu2_experts_keep_single_projection_and_scale_routes_once(self):
        """Non-gated NVFP4 experts must not split ReLU2 or rescale materialized routes."""
        method = XPUModelOptNvFp4MoEMethod(config())
        layer = torch.nn.Module()
        layer.moe_tp_size = layer.moe_ep_size = 1
        layer.moe_runner_config = MoeRunnerConfig(
            activation="relu2", is_gated=False, routed_scaling_factor=2.5
        )
        method.create_weights(
            layer=layer,
            num_experts=3,
            hidden_size=32,
            intermediate_size_per_partition=16,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )
        self.assertEqual(layer.w13_weight.shape, (3, 16, 16))
        self.assertEqual(layer.w13_weight_scale_2.shape, (3, 1))
        up_ref, down_ref = [], []
        for expert in range(3):
            packed, scales, reference = fixture(16, 32, expert)
            multiplier = 0.125 * (expert + 1)
            layer.w13_weight.data[expert].copy_(packed)
            layer.w13_weight_scale.data[expert].copy_(scales)
            layer.w13_weight_scale_2.data[expert].fill_(multiplier)
            up_ref.append((reference * multiplier).to(torch.bfloat16))
            packed, scales, reference = fixture(32, 16, expert + 3)
            multiplier = 0.25 * (expert + 1)
            layer.w2_weight.data[expert].copy_(packed)
            layer.w2_weight_scale.data[expert].copy_(scales)
            layer.w2_weight_scale_2.data[expert] = multiplier
            down_ref.append((reference * multiplier).to(torch.bfloat16))
        layer.to("xpu")
        method.create_moe_runner(layer, layer.moe_runner_config)
        method.process_weights_after_loading(layer)
        x = torch.linspace(-0.2, 0.2, 96).reshape(3, 32).to(torch.bfloat16)
        ids = torch.tensor([[0, 2], [1, 0], [2, -1]])
        weights = torch.tensor([[0.3, 0.7], [0.25, 0.75], [1.0, 0.0]])
        expected = torch.zeros(3, 32)
        for row in range(3):
            for slot in range(2):
                expert = ids[row, slot].item()
                if expert < 0:
                    continue
                up = F.linear(x[row], up_ref[expert])
                expected[row] += (
                    F.linear(F.relu(up).square(), down_ref[expert]).float()
                    * weights[row, slot]
                    * 2.5
                )
        dispatch = StandardDispatchOutput(
            hidden_states=x.to("xpu"),
            hidden_states_scale=None,
            topk_output=SimpleNamespace(
                topk_ids=ids.to("xpu"), topk_weights=(weights * 2.5).to("xpu")
            ),
        )
        actual = method.apply(layer, dispatch).hidden_states
        torch.testing.assert_close(
            actual.cpu().float(),
            expected.to(torch.bfloat16).float(),
            rtol=2e-2,
            atol=2e-3,
        )
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
