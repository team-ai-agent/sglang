import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.models.transformers import sglang_flash_attention_forward
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


class TestTransformersAttentionLayout(unittest.TestCase):
    def test_prefill_decode_layout_and_per_head_gate(self):
        torch.manual_seed(42)
        for dtype in (torch.float32, torch.bfloat16):
            for tokens in (1, 6):
                for scaling in (None, 0.75):
                    with self.subTest(dtype=dtype, tokens=tokens, scaling=scaling):
                        heads, kv_heads, dim = 16, 2, 8
                        query = torch.randn(1, heads, tokens, dim, dtype=dtype)
                        key = torch.randn(1, kv_heads, tokens, dim, dtype=dtype)
                        value = torch.randn_like(key)
                        expected = torch.randn(tokens, heads * dim * 2, dtype=dtype)[
                            :, ::2
                        ]
                        batch = object()

                        def forward(q, k, v, forward_batch):
                            self.assertIs(forward_batch, batch)
                            for actual, source in ((q, query), (k, key), (v, value)):
                                torch.testing.assert_close(
                                    actual, source.transpose(1, 2).reshape(tokens, -1)
                                )
                            return expected

                        attention = SimpleNamespace(
                            scaling=0.125, forward=Mock(side_effect=forward)
                        )
                        output, weights = sglang_flash_attention_forward(
                            SimpleNamespace(layer_idx=3),
                            query,
                            key,
                            value,
                            None,
                            scaling=scaling,
                            attention_instances={3: attention},
                            forward_batch=batch,
                        )
                        self.assertEqual(output.shape, (1, tokens, heads, dim))
                        self.assertIsNone(weights)
                        self.assertEqual(
                            attention.scaling, 0.125 if scaling is None else scaling
                        )
                        attention.forward.assert_called_once()
                        torch.testing.assert_close(output.reshape(tokens, -1), expected)
                        gate = torch.rand(1, tokens, heads, 1, dtype=dtype)
                        torch.testing.assert_close(
                            (output * gate).reshape(tokens, -1),
                            (expected.reshape(1, tokens, heads, dim) * gate).reshape(
                                tokens, -1
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
