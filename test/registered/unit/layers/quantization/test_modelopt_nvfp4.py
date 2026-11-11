import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestModelOptNvfp4(CustomTestCase):
    def test_xpu_merged_checkpoint_scalar_initializes_every_projection(self):
        """Real XPU method selection must use the fused scalar loader, not leave slots unwritten."""
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            packed_modules_mapping={},
            use_per_token_activation=False,
        )
        with patch(
            "sglang.srt.layers.quantization.modelopt_quant.is_xpu", return_value=True
        ):
            layer = MergedColumnParallelLinear(
                input_size=16,
                output_sizes=[16, 16],
                params_dtype=torch.bfloat16,
                quant_config=config,
                parallel_group="replicated",
                prefix="linear",
            )
        for param in (layer.weight_scale_2, layer.input_scale):
            param.data.fill_(float("nan"))
            param.weight_loader(param, torch.tensor(0.125))
            torch.testing.assert_close(param.data, torch.full((2,), 0.125))
        layer.quant_method.process_weights_after_loading(layer)

    def test_serialized_nvfp4_selects_xpu_and_preserves_exclusions(self):
        """A uniform NVFP4 export must reach XPU methods without quantizing excluded layers."""
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            _fuses_routed_scaling_factor_in_topk,
        )

        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["excluded"],
            packed_modules_mapping={},
            use_per_token_activation=False,
        )
        experts = FusedMoE.__new__(FusedMoE)
        nn.Module.__init__(experts)
        with patch(
            "sglang.srt.layers.quantization.modelopt_quant.is_xpu", return_value=True
        ):
            dense = config.get_quant_method(self._make_layer(), "linear")
            moe = config.get_quant_method(experts, "experts")
            excluded = config.get_quant_method(self._make_layer(), "excluded")
            self.assertIsNone(config.get_quant_method(experts, "excluded"))
        self.assertEqual(type(dense).__name__, "XPUModelOptNvFp4A16LinearMethod")
        self.assertEqual(type(moe).__name__, "XPUModelOptNvFp4MoEMethod")
        self.assertEqual(type(excluded).__name__, "UnquantizedLinearMethod")
        self.assertTrue(_fuses_routed_scaling_factor_in_topk(moe, None))

    def test_xpu_dense_rejects_invalid_checkpoint_global_multipliers(self):
        from sglang.srt.hardware_backend.xpu.quantization.modelopt import (
            XPUModelOptNvFp4A16LinearMethod,
        )

        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=False,
        )
        for value in (0.0, float("nan")):
            with self.subTest(value=value):
                layer = nn.Module()
                method = XPUModelOptNvFp4A16LinearMethod(config)
                method.create_weights(
                    layer=layer,
                    input_size_per_partition=16,
                    output_partition_sizes=[16],
                    input_size=16,
                    output_size=16,
                    params_dtype=torch.bfloat16,
                    weight_loader=default_weight_loader,
                )
                layer.weight_scale_2.data.fill_(value)
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    method.process_weights_after_loading(layer)

    def test_xpu_direct_experts_reject_sharded_execution(self):
        from sglang.srt.hardware_backend.xpu.quantization.modelopt import (
            XPUModelOptNvFp4MoEMethod,
        )
        from sglang.srt.layers.moe import MoeRunnerConfig

        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=False,
        )
        method = XPUModelOptNvFp4MoEMethod(config)
        layer = nn.Module()
        layer.moe_tp_size, layer.moe_ep_size = 2, 1
        with self.assertRaisesRegex(NotImplementedError, "TP=1 and EP=1"):
            method.create_moe_runner(layer, MoeRunnerConfig())

    def test_mixed_w4a16_selects_xpu_without_cuda_marlin(self):
        """A serialized ModelOpt W4A16 expert must not require a CUDA platform."""
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "packed_modules_mapping": {},
                "quantized_layers": {
                    "linear": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
                    "linear_nvfp4": {"quant_algo": "NVFP4", "group_size": 16},
                    "experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
                },
            }
        )
        experts = FusedMoE.__new__(FusedMoE)
        nn.Module.__init__(experts)
        with patch(
            "sglang.srt.layers.quantization.modelopt_quant.is_xpu",
            return_value=True,
            create=True,
        ):
            dense = config.get_quant_method(self._make_layer(), "linear")
            native_fp4 = config.get_quant_method(self._make_layer(), "linear_nvfp4")
            moe = config.get_quant_method(experts, "experts")
        self.assertEqual(type(dense).__name__, "XPUModelOptNvFp4A16LinearMethod")
        self.assertEqual(type(native_fp4).__name__, "XPUModelOptNvFp4A16LinearMethod")
        self.assertEqual(type(moe).__name__, "XPUModelOptNvFp4MoEMethod")
        with patch(
            "sglang.srt.layers.quantization.modelopt_quant.is_xpu",
            return_value=True,
        ):
            head = config.get_quant_method(self._make_layer(), "lm_head")
        self.assertEqual(type(head).__name__, "UnquantizedLinearMethod")

    def test_mixed_w4a16_keeps_cuda_method_selection(self):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4A16LinearMethod,
            ModelOptNvFp4FusedMoEMethod,
        )

        config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "packed_modules_mapping": {},
                "quantized_layers": {
                    "linear": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
                    "linear_nvfp4": {"quant_algo": "NVFP4", "group_size": 16},
                    "experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
                },
            }
        )
        experts = FusedMoE.__new__(FusedMoE)
        nn.Module.__init__(experts)
        with (
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.is_xpu",
                return_value=False,
                create=True,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.get_platform",
                return_value=SimpleNamespace(is_blackwell=True),
            ),
        ):
            self.assertIs(
                type(config.get_quant_method(self._make_layer(), "linear")),
                ModelOptNvFp4A16LinearMethod,
            )
            self.assertIs(
                type(config.get_quant_method(self._make_layer(), "linear_nvfp4")),
                ModelOptFp4LinearMethod,
            )
            self.assertIs(
                type(config.get_quant_method(experts, "experts")),
                ModelOptNvFp4FusedMoEMethod,
            )

    def test_xpu_weight_only_rejects_unimplemented_awq_prescale(self):
        from sglang.srt.hardware_backend.xpu.quantization.modelopt import (
            XPUModelOptNvFp4A16LinearMethod,
        )

        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=False,
            is_awq=True,
        )
        with self.assertRaisesRegex(NotImplementedError, "AWQ"):
            XPUModelOptNvFp4A16LinearMethod(config)

    def _make_layer(self):
        return MergedColumnParallelLinear(
            input_size=16,
            output_sizes=[16, 16],
            bias=False,
            parallel_group="replicated",
        )

    def _make_qkv_layer(self):
        return QKVParallelLinear(
            hidden_size=16,
            head_size=8,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            parallel_group="replicated",
        )

    def test_fused_scalar_scale_load_fills_all_logical_slots(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.25, dtype=torch.float32))

        torch.testing.assert_close(scale, torch.tensor([0.25, 0.25]))

    def test_fused_scalar_scale_load_rejects_non_scalar(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        with self.assertRaisesRegex(ValueError, "Expected scalar scale"):
            layer.weight_loader_v2(scale, torch.tensor([0.25, 0.5]))

    def test_fused_qkv_scalar_scale_load_fills_all_logical_slots(self):
        layer = self._make_qkv_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(3, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.125, dtype=torch.float32))

        torch.testing.assert_close(scale, torch.tensor([0.125, 0.125, 0.125]))

    def test_explicit_shard_scale_loads_stay_independent(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.25, dtype=torch.float32), 0)
        layer.weight_loader_v2(scale, torch.tensor(0.5, dtype=torch.float32), 1)

        torch.testing.assert_close(scale, torch.tensor([0.25, 0.5]))

    def test_missing_input_scale_defaults_to_one_and_checkpoint_overwrites(self):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=False,
        )
        layer = nn.Module()
        ModelOptFp4LinearMethod(config).create_weights(
            layer,
            input_size_per_partition=16,
            output_partition_sizes=[16],
            input_size=16,
            output_size=16,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )

        torch.testing.assert_close(layer.input_scale, torch.ones(1))
        default_weight_loader(layer.input_scale, torch.tensor(0.25))
        torch.testing.assert_close(layer.input_scale, torch.tensor([0.25]))

    @patch(
        "sglang.srt.layers.quantization.modelopt_quant.envs."
        "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.get",
        return_value=True,
    )
    def test_modelopt_fp4_per_token_activation_contract(self, _):
        # Serialized ModelOpt FP4 retains the existing environment-controlled
        # per-token activation path.
        serialized_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        # Online modelopt_fp4 always uses per-tensor activation scaling, even
        # when the serialized-checkpoint environment switch is enabled.
        online_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=False,
            group_size=16,
        )

        self.assertTrue(serialized_config.use_per_token_activation)
        self.assertFalse(online_config.use_per_token_activation)
        # nvfp4_online is the public interface for online per-token scaling.
        with self.assertRaisesRegex(ValueError, "Use nvfp4_online"):
            ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=False,
                group_size=16,
                use_per_token_activation=True,
            )

    def test_shared_expert_fusion_requires_matching_fp4_precision(self):
        quantized_shared = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        bf16_shared = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.*.mlp.shared_experts*"],
        )

        gate_only_bf16 = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.*.mlp.shared_expert_gate"],
        )

        self.assertTrue(quantized_shared.can_fuse_shared_expert())
        self.assertFalse(bf16_shared.can_fuse_shared_expert())
        # Only the gate is BF16 (Qwen3-Next NVFP4): the FP4 body still fuses.
        self.assertTrue(gate_only_bf16.can_fuse_shared_expert())


if __name__ == "__main__":
    unittest.main()
