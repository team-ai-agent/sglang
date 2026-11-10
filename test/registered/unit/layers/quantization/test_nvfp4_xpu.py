import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def nvfp4_config(targets):
    weight = {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "tensor_group",
        "group_size": 16,
        "scale_dtype": "torch.float8_e4m3fn",
        "dynamic": False,
    }
    return {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": targets,
                "weights": weight,
                "input_activations": {**weight, "dynamic": "local"},
            }
        },
        "ignore": ["lm_head"],
    }


class TestNVFP4XPUDispatch(CustomTestCase):
    def test_invalid_global_scale_is_not_silently_clamped(self):
        from sglang.srt.hardware_backend.xpu.quantization.nvfp4 import (
            XPUCompressedTensorsW4A16Fp4,
        )

        layer = torch.nn.Module()
        layer.weight_global_scale = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            XPUCompressedTensorsW4A16Fp4().process_weights_after_loading(layer)

    def test_mismatched_fused_global_scales_are_not_approximated(self):
        from sglang.srt.hardware_backend.xpu.quantization.nvfp4 import (
            XPUCompressedTensorsW4A16Fp4,
        )

        layer = torch.nn.Module()
        layer.weight_global_scale = torch.nn.Parameter(torch.tensor([2., 3.]), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "shared global scale"):
            XPUCompressedTensorsW4A16Fp4().process_weights_after_loading(layer)

    def test_direct_expert_path_rejects_unimplemented_parallel_layout(self):
        from types import SimpleNamespace

        from sglang.srt.hardware_backend.xpu.quantization.nvfp4 import (
            XPUCompressedTensorsW4A16Nvfp4MoE,
        )
        from sglang.srt.layers.moe import MoeRunnerConfig

        with self.assertRaisesRegex(NotImplementedError, "TP=1 and EP=1"):
            XPUCompressedTensorsW4A16Nvfp4MoE().create_moe_runner(
                SimpleNamespace(moe_tp_size=2, moe_ep_size=1), MoeRunnerConfig()
            )

    def test_non_xpu_weight_only_scheme_is_not_replaced(self):
        config = CompressedTensorsConfig.from_config(nvfp4_config(["Linear"]))
        weight = next(iter(config.target_scheme_map.values()))["weights"]
        with patch.object(compressed_tensors, "_is_xpu", False):
            scheme = config._get_scheme_from_parts(weight, None)
        self.assertEqual(type(scheme).__name__, "CompressedTensorsW4A16Fp4")

    def test_xpu_dense_does_not_require_cuda_marlin_or_blackwell(self):
        """Real NVFP4 shared projections must get an implemented XPU scheme."""
        config = CompressedTensorsConfig.from_config(nvfp4_config(["Linear"]))
        with patch.object(compressed_tensors, "_is_xpu", True):
            scheme = config.get_linear_scheme(
                torch.nn.Linear(32, 16, bias=False),
                "model.layers.1.mlp.shared_expert.gate_proj",
            )
        self.assertTrue(type(scheme).__name__.startswith("XPU"))

    def test_xpu_experts_do_not_select_a_cuda_only_runner(self):
        """A NVFP4 MoE must retain a native XPU packed-weight execution path."""
        config = CompressedTensorsConfig.from_config(nvfp4_config(["re:.*experts.*"]))
        with (
            patch.object(compressed_tensors, "_is_xpu", True),
            patch.object(compressed_tensors, "_is_npu", False),
        ):
            scheme = config.get_moe_scheme(
                torch.nn.Module(), "model.layers.1.mlp.experts"
            )
        self.assertTrue(type(scheme).__name__.startswith("XPU"))


if __name__ == "__main__":
    unittest.main()
