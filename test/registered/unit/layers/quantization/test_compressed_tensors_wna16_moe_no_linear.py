import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16MoE,
    CompressedTensorsWNA16TritonMoE,
)
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_WNA16_MOE_SCHEMES = (CompressedTensorsWNA16MoE, CompressedTensorsWNA16TritonMoE)
EXPERTS_LAYER = "model.layers.0.mlp.experts"
PER_LAYER_EXPERT_TARGETS = [
    f"{EXPERTS_LAYER}.0.gate_proj",
    f"{EXPERTS_LAYER}.0.up_proj",
    f"{EXPERTS_LAYER}.0.down_proj",
]


def _make_wna16_moe_config(targets, num_bits, **weight_overrides):
    weights = {
        "num_bits": num_bits,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
    }
    weights.update(weight_overrides)
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": targets,
                "weights": weights,
                "input_activations": None,
            }
        },
        "ignore": ["lm_head", "re:.*self_attn.*", "re:.*mlp.gate$"],
    }


@mock.patch.object(compressed_tensors, "_is_xpu", False)
class TestWNA16MoENoLinearGroup(CustomTestCase):
    def _assert_wna16_moe(self, config_dict, expected_bits):
        quant_config = CompressedTensorsConfig.from_config(config_dict)
        self.assertNotIn("Linear", quant_config.target_scheme_map)

        layer = torch.nn.Module()
        scheme = quant_config.get_moe_scheme(layer, layer_name=EXPERTS_LAYER)

        self.assertIsInstance(scheme, _WNA16_MOE_SCHEMES)
        self.assertEqual(scheme.num_bits, expected_bits)
        self.assertEqual(scheme.group_size, 128)

    def test_regex_expert_targets_int4(self):
        config = _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=4)
        self._assert_wna16_moe(config, expected_bits=4)

    def test_regex_expert_targets_int8(self):
        config = _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=8)
        self._assert_wna16_moe(config, expected_bits=8)

    def test_xpu_mixed_int4_int8_experts_do_not_select_cuda_marlin(self):
        """Mixed checkpoint groups must retain their bit width on the XPU path."""
        for bits in (4, 8):
            quant_config = CompressedTensorsConfig.from_config(
                _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=bits)
            )
            with (
                mock.patch.object(compressed_tensors, "_is_xpu", True),
                mock.patch.object(compressed_tensors, "_is_npu", False),
                mock.patch.object(compressed_tensors, "_is_hip", False),
                mock.patch.object(
                    compressed_tensors,
                    "get_moe_runner_backend",
                    return_value=MoeRunnerBackend.AUTO,
                ),
            ):
                scheme = quant_config.get_moe_scheme(
                    torch.nn.Module(), layer_name=EXPERTS_LAYER
                )
            self.assertIsInstance(scheme, CompressedTensorsWNA16TritonMoE)
            self.assertEqual(scheme.num_bits, bits)

    def test_int8_packed_projection_is_not_reinterpreted_as_int4(self):
        """8-bit packed bytes must reach the W8A16 kernel, never the nibble kernel."""
        scheme = CompressedTensorsWNA16TritonMoE.__new__(
            CompressedTensorsWNA16TritonMoE
        )
        scheme.num_bits, scheme.group_size = 8, 128
        layer = torch.nn.Module()
        packed = torch.arange(256, dtype=torch.int64).to(torch.uint8).reshape(1, 2, 128)
        layer.w13_weight_packed = torch.nn.Parameter(
            packed.view(torch.int32).transpose(1, 2).contiguous(), requires_grad=False
        )
        layer.w2_weight_packed = torch.nn.Parameter(
            layer.w13_weight_packed.detach().clone(), requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.ones(1, 1, 2), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.ones(1, 1, 2), requires_grad=False
        )
        scheme.process_weights_after_loading(layer)
        info = scheme.get_triton_quant_info(layer)
        self.assertTrue(info.use_int8_w8a16)
        self.assertFalse(info.use_int4_w4a16)
        torch.testing.assert_close(info.w13_weight, packed)

    def test_per_layer_fqn_expert_targets_int4(self):
        config = _make_wna16_moe_config(PER_LAYER_EXPERT_TARGETS, num_bits=4)
        self._assert_wna16_moe(config, expected_bits=4)

    def test_blackwell_int4_auto_uses_triton(self):
        for group_size in (32, 128):
            with self.subTest(group_size=group_size):
                quant_config = CompressedTensorsConfig.from_config(
                    _make_wna16_moe_config(
                        ["re:.*mlp.experts.*"],
                        num_bits=4,
                        group_size=group_size,
                    )
                )

                with (
                    mock.patch.object(
                        compressed_tensors,
                        "get_moe_runner_backend",
                        return_value=MoeRunnerBackend.AUTO,
                    ),
                    override_platform(is_sm100=True),
                ):
                    scheme = quant_config.get_moe_scheme(
                        torch.nn.Module(), layer_name=EXPERTS_LAYER
                    )

                self.assertIsInstance(scheme, CompressedTensorsWNA16TritonMoE)

    def test_blackwell_auto_rejects_unvalidated_triton_layouts(self):
        cases = {
            "asymmetric": {"symmetric": False},
            "channel": {"strategy": "channel", "group_size": None},
            "group64": {"group_size": 64},
            "actorder": {"actorder": "weight"},
        }
        for name, overrides in cases.items():
            with self.subTest(name=name):
                quant_config = CompressedTensorsConfig.from_config(
                    _make_wna16_moe_config(
                        ["re:.*mlp.experts.*"], num_bits=4, **overrides
                    )
                )
                with (
                    mock.patch.object(
                        compressed_tensors,
                        "get_moe_runner_backend",
                        return_value=MoeRunnerBackend.AUTO,
                    ),
                    override_platform(is_sm100=True),
                ):
                    scheme = quant_config.get_moe_scheme(
                        torch.nn.Module(), layer_name=EXPERTS_LAYER
                    )

                self.assertIsInstance(scheme, CompressedTensorsWNA16MoE)
                self.assertNotIsInstance(scheme, CompressedTensorsWNA16TritonMoE)

    def test_explicit_triton_rejects_unvalidated_layout(self):
        quant_config = CompressedTensorsConfig.from_config(
            _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=4, symmetric=False)
        )

        with (
            mock.patch.object(
                compressed_tensors,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.TRITON,
            ),
            self.assertRaisesRegex(ValueError, "only supports symmetric INT4"),
        ):
            quant_config.get_moe_scheme(torch.nn.Module(), layer_name=EXPERTS_LAYER)

    def test_blackwell_explicit_marlin_is_preserved(self):
        quant_config = CompressedTensorsConfig.from_config(
            _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=4)
        )

        with (
            mock.patch.object(
                compressed_tensors,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.MARLIN,
            ),
            override_platform(is_sm100=True),
        ):
            scheme = quant_config.get_moe_scheme(
                torch.nn.Module(), layer_name=EXPERTS_LAYER
            )

        self.assertIsInstance(scheme, CompressedTensorsWNA16MoE)

    def test_blackwell_int8_auto_keeps_marlin(self):
        quant_config = CompressedTensorsConfig.from_config(
            _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=8)
        )

        with (
            mock.patch.object(
                compressed_tensors,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.AUTO,
            ),
            override_platform(is_sm100=True),
        ):
            scheme = quant_config.get_moe_scheme(
                torch.nn.Module(), layer_name=EXPERTS_LAYER
            )

        self.assertIsInstance(scheme, CompressedTensorsWNA16MoE)
        self.assertNotIsInstance(scheme, CompressedTensorsWNA16TritonMoE)

    def test_asymmetric_group_actorder_loads_full_w2_zero_points(self):
        quant_config = CompressedTensorsConfig.from_config(
            _make_wna16_moe_config(
                ["re:.*mlp.experts.*"],
                num_bits=4,
                symmetric=False,
                actorder="group",
            )
        )
        with mock.patch.object(
            compressed_tensors,
            "get_moe_runner_backend",
            return_value=MoeRunnerBackend.MARLIN,
        ):
            scheme = quant_config.get_moe_scheme(
                torch.nn.Module(), layer_name=EXPERTS_LAYER
            )

        layer = torch.nn.Module()
        layer.moe_tp_size = 2
        scheme.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=256,
            intermediate_size_per_partition=128,
            params_dtype=torch.float16,
        )

        self.assertEqual(layer.w2_weight_scale.shape, (2, 2, 256))
        self.assertEqual(layer.w2_weight_zero_point.shape, (2, 2, 32))
        self.assertTrue(layer.w2_weight_scale.load_full_w2)
        self.assertTrue(layer.w2_weight_zero_point.load_full_w2)


if __name__ == "__main__":
    unittest.main()
