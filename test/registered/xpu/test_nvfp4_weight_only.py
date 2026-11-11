import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.xpu.quantization.nvfp4 import (
    XPUCompressedTensorsW4A16Fp4,
    XPUCompressedTensorsW4A16Nvfp4MoE,
)
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=15, suite="stage-b-test-1-gpu-xpu")


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


@unittest.skipUnless(torch.xpu.is_available(), "requires Intel XPU")
class TestNVFP4WeightOnlyXPU(CustomTestCase):
    def test_dense_preserves_e2m1_block_scale_and_divisor(self):
        """All16 FP4 codes, FP8 block scales and the global divisor survive XPU execution."""
        packed, scales, reference = fixture(16, 32, 0)
        layer = torch.nn.Module()
        layer.weight_packed = torch.nn.Parameter(packed.to("xpu"), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scales.to("xpu"), requires_grad=False)
        layer.weight_global_scale = torch.nn.Parameter(
            torch.tensor([3.0, 3.0], device="xpu"), requires_grad=False
        )
        layer.input_global_scale = torch.nn.Parameter(
            torch.ones(2, device="xpu"), requires_grad=False
        )
        scheme = XPUCompressedTensorsW4A16Fp4(has_input_global_scale=True)
        scheme.process_weights_after_loading(layer)
        x = torch.linspace(-1, 1, 96).reshape(3, 32).to(torch.bfloat16)
        expected = F.linear(x, (reference / 3).to(torch.bfloat16))
        actual = scheme.apply_weights(layer, x.to("xpu"))
        torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-3
        )
        self.assertEqual(layer.weight_packed.dtype, torch.uint8)

    def test_routed_experts_keep_weights_packed_and_skip_dropped_routes(self):
        """Expert-local scales and router weights must not leak between requests or projections."""
        layer = torch.nn.Module()
        layer.moe_tp_size = layer.moe_ep_size = 1
        references_up, references_down = [], []
        packed_up, packed_down, scales_up, scales_down = [], [], [], []
        for expert in range(3):
            packed, scales, reference = fixture(32, 32, expert)
            packed_up.append(packed)
            scales_up.append(scales)
            references_up.append((reference / (expert + 2)).to(torch.bfloat16))
            packed, scales, reference = fixture(32, 16, expert + 3)
            packed_down.append(packed)
            scales_down.append(scales)
            references_down.append((reference / (expert + 4)).to(torch.bfloat16))
        for name, value in {
            "w13_weight_packed": torch.stack(packed_up),
            "w2_weight_packed": torch.stack(packed_down),
            "w13_weight_scale": torch.stack(scales_up),
            "w2_weight_scale": torch.stack(scales_down),
            "w13_weight_global_scale": torch.tensor(
                [[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
            ),
            "w2_weight_global_scale": torch.tensor([4.0, 5.0, 6.0]),
        }.items():
            layer.register_parameter(
                name, torch.nn.Parameter(value.to("xpu"), requires_grad=False)
            )
        scheme = XPUCompressedTensorsW4A16Nvfp4MoE()
        scheme.create_moe_runner(layer, MoeRunnerConfig(routed_scaling_factor=0.75))
        scheme.process_weights_after_loading(layer)
        x = torch.linspace(-0.1, 0.1, 96).reshape(3, 32).to(torch.bfloat16)
        ids = torch.tensor([[0, 2], [1, 0], [2, -1]])
        weights = torch.tensor([[0.3, 0.7], [0.25, 0.75], [1.0, 0.0]])
        expected = torch.zeros(3, 32)
        for row in range(3):
            for slot in range(2):
                expert = ids[row, slot].item()
                if expert < 0:
                    continue
                gate, up = F.linear(x[row], references_up[expert]).chunk(2)
                value = F.linear(F.silu(gate) * up, references_down[expert])
                expected[row] += value.float() * weights[row, slot] * 0.75
        dispatch = StandardDispatchOutput(
            hidden_states=x.to("xpu"),
            hidden_states_scale=None,
            topk_output=SimpleNamespace(
                topk_ids=ids.to("xpu"), topk_weights=weights.to("xpu")
            ),
        )
        actual = scheme.apply_weights(layer, dispatch).hidden_states
        torch.testing.assert_close(
            actual.cpu().float(),
            expected.to(torch.bfloat16).float(),
            rtol=2e-2,
            atol=2e-3,
        )
        self.assertEqual(layer.w13_weight_packed.dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
