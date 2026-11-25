import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.models.gigachat35 import GigaChat35AttentionMLA, deepseek_v2


class TestGigaChatMlaWeights(unittest.TestCase):
    def make_attention(self, cpu=True):
        installed = PackWeightMethod(
            weight_names=["w_kc", "w_vc"], transpose_dims=[[1, 2], [1, 2]]
        )

        def init_base(attention, **kwargs):
            nn.Module.__init__(attention)
            attention.quant_method = installed
            attention.num_heads = 4
            attention.num_local_heads = 4
            attention.v_head_dim = 32
            attention.use_deep_gemm_bmm = False

        with (
            patch.object(deepseek_v2.DeepseekV2AttentionMLA, "__init__", init_base),
            patch("sglang.srt.models.gigachat35.is_cpu", return_value=cpu, create=True),
        ):
            attention = GigaChat35AttentionMLA(
                config=PretrainedConfig(),
                hidden_size=128,
                num_heads=4,
                qk_nope_head_dim=32,
                qk_rope_head_dim=16,
                v_head_dim=32,
                q_lora_rank=64,
                kv_lora_rank=64,
            )
        return attention, installed

    def test_cpu_absorb_weights_keep_unfused_layout(self):
        attention, _ = self.make_attention()
        self.assertIsNone(attention.quant_method)
        generator = torch.Generator().manual_seed(42)
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                attention.w_kc = torch.randn(
                    4, 32, 64, generator=generator
                ).to(dtype)
                attention.w_vc = torch.randn(
                    4, 64, 32, generator=generator
                ).to(dtype)
                attention.w_scale = torch.tensor(0.5)
                query = torch.randn(3, 4, 32, generator=generator).bfloat16()
                values = torch.randn(3, 4, 64, generator=generator).bfloat16()
                scale = attention.w_scale if dtype == torch.float8_e4m3fn else 1
                key_weight = attention.w_kc.bfloat16() * scale
                value_weight = attention.w_vc.bfloat16() * scale
                absorbed = torch.bmm(query.transpose(0, 1), key_weight)
                self.assertEqual(absorbed.shape, (4, 3, 64))
                expected = torch.bmm(
                    values.transpose(0, 1), value_weight
                ).transpose(0, 1).flatten(1, 2)
                actual = attention.project_mla_values(values, None)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                with self.assertRaisesRegex(RuntimeError, "Expected size"):
                    torch.bmm(query.transpose(0, 1), key_weight.transpose(1, 2))

    def test_accelerator_quant_method_is_unchanged(self):
        attention, installed = self.make_attention(cpu=False)
        self.assertIs(attention.quant_method, installed)

    def test_gated_mla_still_dispatches_unfused(self):
        attention, _ = self.make_attention()
        methods = deepseek_v2.AttnForwardMethod
        batch = SimpleNamespace()
        for method in (methods.MLA_FUSED_ROPE_CPU, methods.MLA_FUSED_ROPE_ROCM):
            with (
                self.subTest(method=method),
                patch.object(
                    deepseek_v2.DeepseekV2AttentionMLA,
                    "dispatch_attn_forward_method",
                    return_value=method,
                ),
            ):
                self.assertEqual(
                    attention.dispatch_attn_forward_method(batch), methods.MLA
                )


if __name__ == "__main__":
    unittest.main()
