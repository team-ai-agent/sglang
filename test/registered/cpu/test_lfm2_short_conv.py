import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.lfm2 import Lfm2ShortConv
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


class TupleLinear(nn.Module):
    def forward(self, x):
        return x, None


class TestLfm2ShortConvCPU(unittest.TestCase):
    def make_layer(self, dtype, bias):
        layer = Lfm2ShortConv.__new__(Lfm2ShortConv)
        nn.Module.__init__(layer)
        layer.layer_idx = 0
        layer.hidden_size_per_partition = 7
        layer.conv_weight = nn.Parameter(torch.randn(7, 3, dtype=dtype))
        layer.conv_bias = nn.Parameter(torch.randn(7, dtype=dtype)) if bias else None
        layer.in_proj = TupleLinear()
        layer.out_proj = TupleLinear()
        return layer

    def run_layer(self, layer, x, state, lengths, slots, prefixes, decode=False):
        meta = SimpleNamespace(
            layer_cache=SimpleNamespace(conv=[state]),
            cache_indices=torch.tensor(slots, dtype=torch.int64),
            has_initial_state=torch.tensor(prefixes, dtype=torch.bool),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
            extend_seq_lens_cpu=lengths,
        )
        backend = SimpleNamespace(conv_state_metadata=lambda *_: meta)
        with patch("sglang.srt.models.lfm2.get_attn_backend", return_value=backend):
            return layer(x, batch)

    @torch.no_grad()
    def test_prefill_decode_and_cached_chunks(self):
        for dtype in (torch.float32, torch.bfloat16):
            for bias in (False, True):
                with self.subTest(dtype=dtype, bias=bias):
                    torch.manual_seed(42)
                    layer = self.make_layer(dtype, bias)
                    x = torch.randn(9, 21, dtype=dtype)
                    b, c, value = x.chunk(3, dim=-1)
                    bx = b * value
                    expected = (
                        c
                        * F.conv1d(
                            F.pad(bx.T.unsqueeze(0), (2, 0)),
                            layer.conv_weight.unsqueeze(1),
                            layer.conv_bias,
                            groups=7,
                        )
                        .squeeze(0)
                        .T
                    )
                    for chunks in ([9], [1, 2, 6], [3, 1, 1, 1, 1, 1, 1]):
                        state = torch.randn(4, 7, 2, dtype=dtype)
                        untouched = state[[0, 1, 3]].clone()
                        results = []
                        start = 0
                        for length in chunks:
                            results.append(
                                self.run_layer(
                                    layer,
                                    x[start : start + length],
                                    state,
                                    [length],
                                    [2],
                                    [start > 0],
                                    decode=start > 0 and length == 1,
                                )
                            )
                            start += length
                        torch.testing.assert_close(torch.cat(results), expected)
                        torch.testing.assert_close(state[2], bx[-2:].T)
                        torch.testing.assert_close(state[[0, 1, 3]], untouched)

    @torch.no_grad()
    def test_variable_sequences_prefixes_and_padding(self):
        torch.manual_seed(123)
        layer = self.make_layer(torch.bfloat16, True)
        for decode in (False, True):
            lengths = [1, 1, 1] if decode else [1, 2, 5]
            x = torch.randn(sum(lengths), 21, dtype=torch.bfloat16)
            state = torch.randn(4, 7, 2, dtype=torch.bfloat16)
            expected_state = state.clone()
            expected = torch.zeros(sum(lengths), 7, dtype=torch.bfloat16)
            slots = [2, -1, 0]
            prefixes = [True, False, decode]
            start = 0
            for length, slot, prefix in zip(lengths, slots, prefixes):
                end = start + length
                if slot != -1:
                    b, c, value = x[start:end].chunk(3, dim=-1)
                    history = state[slot] if prefix else torch.zeros_like(state[slot])
                    sequence = torch.cat((history, (b * value).T), dim=-1)
                    # Explicit per-tap FP32 accumulation is independent of conv1d.
                    conv = sum(
                        sequence[:, tap : tap + length].float()
                        * layer.conv_weight[:, tap, None].float()
                        for tap in range(3)
                    )
                    conv += layer.conv_bias[:, None].float()
                    expected[start:end] = c * conv.T.to(c.dtype)
                    expected_state[slot] = sequence[:, -2:]
                start = end
            actual = self.run_layer(
                layer, x, state, lengths, slots, prefixes, decode=decode
            )
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(state, expected_state)

    def test_speculative_verification_rejects_before_state_mutation(self):
        layer = self.make_layer(torch.float32, False)
        state = torch.randn(1, 7, 2)
        before = state.clone()
        with self.assertRaisesRegex(NotImplementedError, "not supported on CPU"):
            layer._forward_cpu(
                torch.randn(1, 7),
                state,
                None,
                SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
            )
        torch.testing.assert_close(state, before)


if __name__ == "__main__":
    unittest.main()
