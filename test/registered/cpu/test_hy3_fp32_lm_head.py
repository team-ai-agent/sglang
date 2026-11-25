import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F

from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.models import hunyuan_v3
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu-intel")


class TestHy3FP32LMHead(unittest.TestCase):
    def test_checkpoint_head_precision_and_cpu_layout(self):
        torch.manual_seed(42)
        checkpoint_weight = torch.randn(32, 64).bfloat16()
        hidden = torch.randn(3, 64).bfloat16()
        for cpu in (True, False):
            for fp32 in (True, False):
                with self.subTest(cpu=cpu, fp32=fp32):
                    config = SimpleNamespace(
                        vocab_size=32,
                        hidden_size=64,
                        enable_lm_head_fp32=fp32,
                        tie_word_embeddings=False,
                    )

                    def make_head(vocab, width, params_dtype=None, **kwargs):
                        head = torch.nn.Linear(
                            width,
                            vocab,
                            bias=False,
                            dtype=params_dtype or torch.bfloat16,
                        )
                        with torch.no_grad():
                            head.weight.copy_(checkpoint_weight)
                        head.quant_method = UnquantizedEmbeddingMethod()
                        if cpu and head.weight.dtype == torch.bfloat16:
                            head.quant_method = PackWeightMethod(
                                weight_names=["weight"]
                            )
                        head.quant_method.process_weights_after_loading(head)
                        return head

                    with (
                        patch.object(hunyuan_v3, "is_cpu", return_value=cpu),
                        patch.object(
                            hunyuan_v3,
                            "get_exec",
                            return_value=SimpleNamespace(
                                features=SimpleNamespace(enable_fp32_lm_head=False)
                            ),
                        ),
                        patch.object(
                            hunyuan_v3, "HYV3Model", return_value=torch.nn.Identity()
                        ),
                        patch.object(
                            hunyuan_v3,
                            "LogitsProcessor",
                            return_value=torch.nn.Identity(),
                        ),
                        patch.object(
                            hunyuan_v3, "ParallelLMHead", side_effect=make_head
                        ),
                    ):
                        model = hunyuan_v3.HYV3ForCausalLM(config)
                    processor = SimpleNamespace(
                        use_fp32_lm_head=fp32,
                        use_fp32_lm_head_output=False,
                        rl_on_policy_target=None,
                    )
                    actual = LogitsProcessor._compute_lm_head(
                        processor, hidden, model.lm_head
                    )
                    expected = F.linear(
                        hidden.float() if fp32 else hidden,
                        checkpoint_weight.float() if fp32 else checkpoint_weight,
                    )
                    torch.testing.assert_close(actual, expected)
                    expected_dtype = torch.float32 if cpu and fp32 else torch.bfloat16
                    self.assertEqual(model.lm_head.weight.dtype, expected_dtype)
                    if cpu and fp32:
                        self.assertFalse(
                            getattr(model.lm_head, "use_intel_amx_backend", False)
                        )

    def test_packed_weight_is_not_an_fp32_matrix(self):
        torch.manual_seed(43)
        weight = torch.randn(32, 64).bfloat16()
        hidden = torch.randn(3, 64).bfloat16()
        packed = torch.ops.sgl_kernel.convert_weight_packed(weight)
        wrong = F.linear(hidden.float(), packed.float())
        expected = F.linear(hidden.float(), weight.float())
        self.assertGreater((wrong - expected).abs().max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
