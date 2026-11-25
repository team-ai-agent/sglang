import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sgl_kernel  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.models import step3p5
from sglang.srt.utils import cpu_has_amx_support
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu-intel")


class _Experts(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.runner_backend = Mock()
        self.routing_method_type = Mock()

    def forward(self, hidden_states, topk_output):
        return torch.zeros_like(hidden_states)


@unittest.skipUnless(cpu_has_amx_support(), "Requires Intel AMX")
class TestStep3p5Fp32Router(unittest.TestCase):
    def test_router_logits_after_cpu_weight_processing(self):
        for need_fp32 in (False, True):
            with self.subTest(need_fp32_gate=need_fp32):
                topk = Mock(
                    side_effect=lambda hidden, logits: StandardTopKOutput(
                        topk_weights=torch.ones(hidden.shape[0], 4),
                        topk_ids=torch.zeros(hidden.shape[0], 4, dtype=torch.int32),
                        router_logits=logits,
                    )
                )
                config = SimpleNamespace(
                    need_fp32_gate=need_fp32,
                    moe_router_scaling_factor=3.0,
                    use_moe_router_bias=True,
                    moe_num_experts=16,
                    swiglu_limits=[0],
                    moe_top_k=4,
                    hidden_size=128,
                    moe_intermediate_size=128,
                )
                with (
                    patch.object(
                        step3p5, "get_parallel",
                        return_value=SimpleNamespace(tp_size=1),
                    ),
                    patch.object(
                        step3p5, "get_exec",
                        return_value=SimpleNamespace(
                            moe=SimpleNamespace(ep_num_redundant_experts=0)
                        ),
                    ),
                    patch.object(
                        step3p5, "get_moe_a2a_backend",
                        return_value=SimpleNamespace(is_deepep=lambda: False),
                    ),
                    patch.object(step3p5, "get_moe_impl_class", return_value=_Experts),
                    patch.object(step3p5, "TopK", return_value=topk),
                    patch("torch.get_default_dtype", return_value=torch.bfloat16),
                ):
                    model = step3p5.Step3p5MoEMLP(config, layer_id=0)
                expected_dtype = torch.float32 if need_fp32 else torch.bfloat16
                self.assertEqual(model.gate.weight.dtype, expected_dtype)
                weight = (
                    torch.arange(16 * 128, dtype=torch.float32)
                    .reshape(16, 128).remainder(17).sub(8).div(16)
                    .to(expected_dtype)
                )
                model.gate.weight.data.copy_(weight)
                model.gate.quant_method.process_weights_after_loading(model.gate)
                inputs = (
                    torch.arange(3 * 128, dtype=torch.float32)
                    .reshape(3, 128).remainder(13).sub(6).div(8)
                    .to(torch.bfloat16)
                )
                model.forward_normal(inputs)
                logits = topk.call_args.args[1]
                expected = F.linear(inputs.to(expected_dtype), weight)
                self.assertEqual(logits.dtype, expected_dtype)
                torch.testing.assert_close(logits, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
