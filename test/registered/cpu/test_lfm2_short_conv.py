import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.lfm2 import Lfm2ShortConv
from sglang.srt.models.lfm2_moe import Lfm2MoeShortConv
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


class TupleLinear(nn.Module):
    def forward(self, x):
        return x, None


@contextlib.contextmanager
def _maybe_disable_cpp_short_conv(use_cpp: bool):
    has_op = hasattr(torch.ops.sgl_kernel, "short_conv1d_cpu")
    if use_cpp or not has_op:
        yield
        return
    saved = torch.ops.sgl_kernel.short_conv1d_cpu
    delattr(torch.ops.sgl_kernel, "short_conv1d_cpu")
    try:
        yield
    finally:
        setattr(torch.ops.sgl_kernel, "short_conv1d_cpu", saved)


class TestLfm2ShortConvCPU(unittest.TestCase):
    def make_layer(self, dtype, bias, dim=7, cls=Lfm2ShortConv):
        layer = cls.__new__(cls)
        nn.Module.__init__(layer)
        layer.layer_idx = 0
        layer.hidden_size_per_partition = dim
        layer.conv_weight = nn.Parameter(torch.randn(dim, 3, dtype=dtype))
        layer.conv_bias = nn.Parameter(torch.randn(dim, dtype=dtype)) if bias else None
        layer.in_proj = TupleLinear()
        layer.out_proj = TupleLinear()
        return layer

    def run_layer(
        self, layer, x, state, lengths, slots, prefixes, decode=False, mod="lfm2"
    ):
        qsl = torch.tensor([0] + list(lengths), dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        meta = SimpleNamespace(
            layer_cache=SimpleNamespace(conv=[state]),
            cache_indices=torch.tensor(slots, dtype=torch.int64),
            has_initial_state=torch.tensor(prefixes, dtype=torch.bool),
            query_start_loc=qsl,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
            extend_seq_lens_cpu=lengths,
        )
        backend = SimpleNamespace(conv_state_metadata=lambda *_: meta)
        with patch(f"sglang.srt.models.{mod}.get_attn_backend", return_value=backend):
            return layer(x, batch)

    @torch.no_grad()
    def test_prefill_decode_and_cached_chunks(self):
        for use_cpp in (True, False):
            for cls, mod in ((Lfm2ShortConv, "lfm2"), (Lfm2MoeShortConv, "lfm2_moe")):
                for dim in (7, 96):
                    for dtype in (torch.float32, torch.bfloat16):
                        for bias in (False, True):
                            with (
                                self.subTest(
                                    use_cpp=use_cpp,
                                    cls=cls.__name__,
                                    dim=dim,
                                    dtype=dtype,
                                    bias=bias,
                                ),
                                _maybe_disable_cpp_short_conv(use_cpp),
                            ):
                                torch.manual_seed(42)
                                layer = self.make_layer(dtype, bias, dim=dim, cls=cls)
                                x = torch.randn(9, 3 * dim, dtype=dtype)
                                b, c, value = x.chunk(3, dim=-1)
                                bx = b * value
                                expected = (
                                    c
                                    * F.conv1d(
                                        F.pad(bx.T.unsqueeze(0), (2, 0)),
                                        layer.conv_weight.unsqueeze(1),
                                        layer.conv_bias,
                                        groups=dim,
                                    )
                                    .squeeze(0)
                                    .T
                                )
                                tol = (
                                    {"rtol": 1.6e-2, "atol": 2e-3}
                                    if dtype == torch.bfloat16
                                    else {"rtol": 1e-5, "atol": 1e-6}
                                )
                                for chunks in (
                                    [9],
                                    [1, 2, 6],
                                    [3, 1, 1, 1, 1, 1, 1],
                                ):
                                    state = torch.randn(4, dim, 2, dtype=dtype)
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
                                                mod=mod,
                                            )
                                        )
                                        start += length
                                    torch.testing.assert_close(
                                        torch.cat(results), expected, **tol
                                    )
                                    torch.testing.assert_close(state[2], bx[-2:].T)
                                    torch.testing.assert_close(
                                        state[[0, 1, 3]], untouched
                                    )

    @torch.no_grad()
    def test_variable_sequences_prefixes_and_padding(self):
        for use_cpp in (True, False):
            with _maybe_disable_cpp_short_conv(use_cpp):
                torch.manual_seed(123)
                layer = self.make_layer(torch.bfloat16, True, dim=70)
                for decode in (False, True):
                    lengths = [1, 1, 1] if decode else [1, 2, 5]
                    x = torch.randn(sum(lengths), 210, dtype=torch.bfloat16)
                    state = torch.randn(4, 70, 2, dtype=torch.bfloat16)
                    expected_state = state.clone()
                    expected = torch.zeros(sum(lengths), 70, dtype=torch.bfloat16)
                    slots = [2, -1, 0]
                    prefixes = [True, False, decode]
                    start = 0
                    for length, slot, prefix in zip(lengths, slots, prefixes):
                        end = start + length
                        if slot != -1:
                            b, c, value = x[start:end].chunk(3, dim=-1)
                            history = (
                                state[slot] if prefix else torch.zeros_like(state[slot])
                            )
                            sequence = torch.cat((history, (b * value).T), dim=-1)
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
