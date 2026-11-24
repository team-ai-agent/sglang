import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    track_mamba_states_all_layers,
    track_mamba_states_if_needed,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.lfm3 import Lfm3ShortConv
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


class TestLfm3ShortConvCPU(unittest.TestCase):
    def make_layer(self, dtype, bias=False, dim=96, kernel=7):
        layer = Lfm3ShortConv.__new__(Lfm3ShortConv)
        nn.Module.__init__(layer)
        layer.layer_idx = 0
        layer.linear_layer_idx = 0
        layer.conv_kernel = kernel
        layer.use_bias = bias
        layer.hidden_size = dim
        layer.hidden_size_per_partition = dim
        layer.conv_weight = nn.Parameter(torch.randn(dim, kernel, dtype=dtype))
        layer.conv_bias = nn.Parameter(torch.randn(dim, dtype=dtype)) if bias else None
        layer.in_proj = TupleLinear()
        layer.out_proj = TupleLinear()
        return layer

    def reference(self, layer, proj):
        b_gate, c_gate, x = proj.chunk(3, dim=-1)
        bx = b_gate * x
        windows = F.pad(bx.T, (layer.conv_kernel - 1, 0)).unfold(
            -1, layer.conv_kernel, 1
        )
        conv = torch.einsum(
            "htk,hk->th", windows.float(), layer.conv_weight.float()
        )
        if layer.conv_bias is not None:
            conv = conv + layer.conv_bias.float()
        return c_gate * conv.to(proj.dtype)

    def run_step(self, layer, conv_state, proj, slots, lengths=None, track_mask=None, track_dst=None):
        decode = lengths is None
        qsl = (
            torch.arange(len(slots) + 1, dtype=torch.int32)
            if decode
            else torch.tensor([0] + list(lengths), dtype=torch.int32).cumsum(
                0, dtype=torch.int32
            )
        )
        metadata = SimpleNamespace(
            mamba_cache_indices=torch.tensor(slots, dtype=torch.int32),
            query_start_loc=qsl,
            track_conv_indices=None,
            conv_states_mask_indices=None,
            mamba_track_indices=(
                torch.tensor(track_dst, dtype=torch.int32)
                if track_dst is not None
                else None
            ),
        )
        layer_cache = SimpleNamespace(
            conv=[conv_state], temporal=torch.empty(conv_state.shape[0], 0)
        )
        pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(mamba2_layer_cache=lambda _: layer_cache)
        )
        backend = SimpleNamespace(
            linear_attn_backend=SimpleNamespace(forward_metadata=metadata)
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
            extend_seq_lens_cpu=lengths,
            mamba_track_mask=(
                torch.tensor(track_mask, dtype=torch.bool)
                if track_mask is not None
                else None
            ),
            batch_size=len(slots),
        )
        with (
            patch("sglang.srt.models.lfm3.get_req_to_token_pool", return_value=pool),
            patch("sglang.srt.models.lfm3.get_attn_backend", return_value=backend),
        ):
            return layer(proj, batch)

    @torch.no_grad()
    def test_lfm3_width7_prefill_decode_and_extra_buffer_tracking(self):
        for use_cpp in (True, False):
            for dtype in (torch.float32, torch.bfloat16):
                for bias in (False, True):
                    for dim in (8, 96):
                        with (
                            self.subTest(
                                use_cpp=use_cpp, dtype=dtype, bias=bias, dim=dim
                            ),
                            _maybe_disable_cpp_short_conv(use_cpp),
                        ):
                            torch.manual_seed(42)
                            layer = self.make_layer(dtype, bias=bias, dim=dim, kernel=7)
                            states = torch.zeros(5, dim, 6, dtype=dtype)
                            seq_a = torch.randn(10, 3 * dim, dtype=dtype)
                            seq_b = torch.randn(8, 3 * dim, dtype=dtype)
                            ref_a = self.reference(layer, seq_a)
                            ref_b = self.reference(layer, seq_b)
                            tol = (
                                {"rtol": 1.6e-2, "atol": 2e-3}
                                if dtype == torch.bfloat16
                                else {"rtol": 1e-5, "atol": 1e-6}
                            )

                            # Chunk 1 prefill (fresh zeroed slots 2 and 0)
                            out1 = self.run_step(
                                layer,
                                states,
                                torch.cat([seq_a[:5], seq_b[:3]], dim=0),
                                [2, 0],
                                [5, 3],
                            )
                            torch.testing.assert_close(
                                out1, torch.cat([ref_a[:5], ref_b[:3]], dim=0), **tol
                            )

                            # Chunk 2 prefill (continuing state)
                            out2 = self.run_step(
                                layer,
                                states,
                                torch.cat([seq_a[5:9], seq_b[3:7]], dim=0),
                                [2, 0],
                                [4, 4],
                            )
                            torch.testing.assert_close(
                                out2, torch.cat([ref_a[5:9], ref_b[3:7]], dim=0), **tol
                            )

                            # Reordered decode with CPU mamba_track_mask snapshot into slot 4
                            out3 = self.run_step(
                                layer,
                                states,
                                torch.cat([seq_b[7:8], seq_a[9:10]], dim=0),
                                [0, 2],
                                lengths=None,
                                track_mask=[True, False],
                                track_dst=[4, 3],
                            )
                            torch.testing.assert_close(
                                out3, torch.cat([ref_b[7:8], ref_a[9:10]], dim=0), **tol
                            )
                            torch.testing.assert_close(states[4], states[0])
                            self.assertEqual(states[3].abs().sum().item(), 0.0)

    def test_cpu_track_mamba_states_helpers(self):
        conv_pool = torch.randn(2, 4, 8, 6, dtype=torch.bfloat16)
        ssm_pool = torch.empty(2, 4, 0, dtype=torch.bfloat16)
        before_1 = conv_pool[:, 1].clone()
        track_mamba_states_if_needed(
            conv_pool[0],
            ssm_pool[0],
            torch.tensor([0, -1], dtype=torch.int32),
            torch.tensor([True, True], dtype=torch.bool),
            torch.tensor([3, 1], dtype=torch.int32),
            batch_size=2,
            check_freed_slots=True,
        )
        torch.testing.assert_close(conv_pool[0, 3], conv_pool[0, 0])
        torch.testing.assert_close(conv_pool[0, 1], before_1[0])

        track_mamba_states_all_layers(
            conv_pool,
            ssm_pool,
            torch.tensor([2, 0], dtype=torch.int32),
            torch.tensor([True, False], dtype=torch.bool),
            torch.tensor([1, 3], dtype=torch.int32),
            batch_size=2,
        )
        torch.testing.assert_close(conv_pool[:, 1], conv_pool[:, 2])


if __name__ == "__main__":
    unittest.main()
