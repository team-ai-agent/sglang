import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sgl_kernel  # noqa: F401
import torch

from sglang.srt.arg_groups.model_overrides import iquest_q1 as overrides
from sglang.srt.models.iquest_q1 import _learned_sink_logits
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-test-cpu-intel")


class TestQueryDependentAttentionSinks(unittest.TestCase):
    def reference(self, query, keys, values, sinks, prefixes, extends, window):
        outputs = []
        start = 0
        group = query.shape[1] // keys[0].shape[1]
        for key, value, prefix, extend in zip(keys, values, prefixes, extends):
            key = key.repeat_interleave(group, dim=1).float()
            value = value.repeat_interleave(group, dim=1).float()
            for row in range(extend):
                position = prefix + row
                begin = max(0, position - window + 1) if window else 0
                q = query[start + row].float()
                scores = torch.einsum("hd,khd->hk", q, key[begin : position + 1])
                scores *= query.shape[-1] ** -0.5
                if sinks is not None:
                    sink = sinks if sinks.ndim == 1 else sinks[start + row]
                    scores = torch.cat([scores, sink.float().unsqueeze(-1)], dim=-1)
                weights = scores.softmax(-1)
                if sinks is not None:
                    weights = weights[..., :-1]
                outputs.append(
                    torch.einsum("hk,khd->hd", weights, value[begin : position + 1])
                )
            start += extend
        return torch.stack(outputs).to(query.dtype)

    def run_native(self, decode):
        torch.manual_seed(73)
        for heads, kv_heads, dim in [(4, 4, 64), (12, 2, 128), (6, 1, 128)]:
            for dtype in (torch.bfloat16, torch.float16):
                for window in (0, 5):
                    for kind in ("none", "static", "dynamic"):
                        with self.subTest(
                            decode=decode,
                            shape=(heads, kv_heads, dim),
                            dtype=dtype,
                            window=window,
                            sinks=kind,
                        ):
                            prefixes = [269, 9]
                            extends = [1, 1] if decode else [7, 3]
                            lengths = [p + n for p, n in zip(prefixes, extends)]
                            keys = [
                                torch.randn(n, kv_heads, dim, dtype=dtype)
                                for n in lengths
                            ]
                            values = [torch.randn_like(k) for k in keys]
                            total = sum(lengths)
                            perm = torch.randperm(total)
                            key_cache = torch.empty(total, kv_heads, dim, dtype=dtype)
                            value_cache = torch.empty_like(key_cache)
                            key_cache[perm] = torch.cat(keys)
                            value_cache[perm] = torch.cat(values)
                            request_map = torch.zeros(
                                2, max(lengths), dtype=torch.int32
                            )
                            request_map[1, : lengths[0]] = perm[: lengths[0]].int()
                            request_map[0, : lengths[1]] = perm[lengths[0] :].int()
                            requests = torch.tensor([1, 0], dtype=torch.int64)
                            seq_lens = torch.tensor(lengths, dtype=torch.int64)
                            query = torch.randn(sum(extends), heads, dim, dtype=dtype)
                            query = query.transpose(0, 1).contiguous().transpose(0, 1)
                            sink = None
                            if kind == "static":
                                sink = torch.randn(heads, dtype=dtype)
                            elif kind == "dynamic":
                                sink = torch.randn(sum(extends), heads * 2)[:, ::2]
                            output = torch.empty(query.shape, dtype=query.dtype)
                            if decode:
                                logits = torch.empty(2, heads, 8, dim + 1)
                                torch.ops.sgl_kernel.decode_attention_cpu(
                                    query,
                                    key_cache,
                                    value_cache,
                                    1.0,
                                    1.0,
                                    output,
                                    None,
                                    None,
                                    torch.empty(0, dtype=torch.int64),
                                    logits,
                                    request_map,
                                    requests,
                                    seq_lens,
                                    dim**-0.5,
                                    0.0,
                                    False,
                                    window,
                                    None,
                                    sink,
                                )
                            else:
                                key_extend = torch.cat(
                                    [k[p:] for k, p in zip(keys, prefixes)]
                                )
                                value_extend = torch.cat(
                                    [v[p:] for v, p in zip(values, prefixes)]
                                )
                                torch.ops.sgl_kernel.extend_attention_cpu(
                                    query,
                                    key_extend,
                                    value_extend,
                                    output,
                                    key_cache,
                                    value_cache,
                                    1.0,
                                    1.0,
                                    request_map,
                                    requests,
                                    seq_lens,
                                    torch.tensor(extends, dtype=torch.int32),
                                    torch.tensor([0, extends[0]], dtype=torch.int32),
                                    max(extends),
                                    dim**-0.5,
                                    0.0,
                                    False,
                                    window,
                                    None,
                                    sink,
                                    None,
                                    True,
                                )
                            expected = self.reference(
                                query, keys, values, sink, prefixes, extends, window
                            )
                            self.assertTrue(torch.isfinite(output).all())
                            torch.testing.assert_close(
                                output, expected, rtol=0.02, atol=0.015
                            )

    def test_extend_static_and_query_dependent_sinks(self):
        self.run_native(decode=False)

    def test_decode_static_and_query_dependent_sinks(self):
        self.run_native(decode=True)

    def test_invalid_sinks_have_explicit_errors(self):
        query = torch.zeros(1, 6, 128, dtype=torch.bfloat16)
        cache = torch.zeros(1, 1, 128, dtype=torch.bfloat16)
        indices = torch.zeros(1, dtype=torch.int64)
        seq_lens = torch.ones(1, dtype=torch.int64)
        request_map = torch.zeros(1, 1, dtype=torch.int32)
        invalid = [
            (torch.zeros(7), "shape"),
            (torch.zeros(2, 6), "shape"),
            (torch.zeros(1, 6, 1), "shape"),
            (torch.zeros(6, dtype=torch.int64), "dtype"),
            (torch.zeros(6, dtype=torch.float64), "dtype"),
            (torch.empty(6, device="meta"), "Meta tensors"),
        ]
        for decode in (False, True):
            for sink, error in invalid:
                with self.subTest(decode=decode, shape=sink.shape, dtype=sink.dtype):
                    with self.assertRaisesRegex(RuntimeError, error):
                        if decode:
                            torch.ops.sgl_kernel.decode_attention_cpu(
                                query,
                                cache,
                                cache.clone(),
                                1.0,
                                1.0,
                                torch.empty_like(query),
                                None,
                                None,
                                indices,
                                torch.empty(1, 6, 8, 129),
                                request_map,
                                indices,
                                seq_lens,
                                128**-0.5,
                                0.0,
                                False,
                                0,
                                None,
                                sink,
                            )
                        else:
                            torch.ops.sgl_kernel.extend_attention_cpu(
                                query,
                                cache,
                                cache,
                                torch.empty_like(query),
                                cache,
                                cache.clone(),
                                1.0,
                                1.0,
                                request_map,
                                indices,
                                seq_lens,
                                torch.ones(1, dtype=torch.int32),
                                torch.zeros(1, dtype=torch.int32),
                                1,
                                128**-0.5,
                                0.0,
                                False,
                                0,
                                None,
                                sink,
                                None,
                                True,
                            )

    def test_learned_sink_logits_preserve_reference_and_geometry(self):
        for dtype in (torch.float32, torch.bfloat16):
            for dim in (128, 80):
                for tokens in (0, 7):
                    query = torch.randn(tokens, 12, dim * 2, dtype=dtype)[..., ::2]
                    key = torch.randn(2, dim * 2)[:, ::2]
                    expected = (
                        query.float()
                        * key.to(dtype).repeat_interleave(6, dim=0).float()
                    ).sum(-1) * dim**-0.5
                    actual = _learned_sink_logits(query, key, dim**-0.5)
                    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=5e-7)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertTrue(actual.is_contiguous())
        with self.assertRaisesRegex(ValueError, "geometry"):
            _learned_sink_logits(torch.zeros(1, 3, 128), torch.zeros(2, 128), 1.0)

    def test_cpu_selection_does_not_relax_gpu_or_speculative_guards(self):
        cfg = SimpleNamespace(
            device="cpu",
            tp_size=4,
            attention_backend=None,
            prefill_attention_backend=None,
            decode_attention_backend=None,
            speculative_algorithm=None,
            speculative_draft_attention_backend=None,
            attn_cp_size=1,
            dcp_size=1,
            enable_prefill_cp=False,
            speculative_eagle_topk=None,
            enable_multi_layer_eagle=False,
            pp_size=1,
        )
        hf_config = SimpleNamespace(num_attention_heads=48, num_key_value_heads=8)
        with (
            patch.object(overrides, "resolving_view", side_effect=lambda x: x),
            patch.object(overrides, "attn_dp_enabled_of", return_value=False),
        ):
            self.assertEqual(
                overrides._iquest_q1_overrides(cfg, hf_config),
                {"attention_backend": "intel_amx"},
            )
            cfg.tp_size = 6
            with self.assertRaisesRegex(ValueError, "must be divisible by TP size"):
                overrides._iquest_q1_overrides(cfg, hf_config)
            cfg.tp_size = 4
            cfg.device = "cuda"
            self.assertEqual(
                overrides._iquest_q1_overrides(cfg, hf_config),
                {"attention_backend": "fa3"},
            )
            cfg.attention_backend = "intel_amx"
            with self.assertRaisesRegex(ValueError, "requires FA3"):
                overrides._iquest_q1_overrides(cfg, hf_config)
            cfg.device = "cpu"
            cfg.speculative_algorithm = "EAGLE"
            with self.assertRaisesRegex(ValueError, "CPU speculative"):
                overrides._iquest_q1_overrides(cfg, hf_config)


if __name__ == "__main__":
    unittest.main()
