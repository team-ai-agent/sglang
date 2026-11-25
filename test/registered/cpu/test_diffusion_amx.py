import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.amx_attn import (
    AMXATTNImpl,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    _cached_get_attn_backend,
    get_attn_backend,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    _split_unquantized_merged_linear,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.cpu import CpuPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import precision
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="extra-test-cpu-intel")

_SELECTOR = "sglang.multimodal_gen.runtime.layers.attention.selector"


class TestCpuAttentionBackendSelection(CustomTestCase):
    """The selector asks the platform once per candidate backend. The CPU
    platform used to answer amx_attn to every question, so a layer whose
    supported set lacks amx_attn failed at startup, an explicit torch_sdpa
    still ran amx_attn, and fp32 layers reached a kernel that only dispatches
    bf16/fp16."""

    def setUp(self) -> None:
        _cached_get_attn_backend.cache_clear()

    def _resolve(
        self,
        *,
        supported: set[AttentionBackendEnum],
        dtype: torch.dtype = torch.bfloat16,
        selected: AttentionBackendEnum | None = None,
    ) -> AttentionBackendEnum:
        with (
            patch(f"{_SELECTOR}.get_global_forced_attn_backend", return_value=None),
            patch(f"{_SELECTOR}.get_component_forced_attn_backend", return_value=None),
            patch(
                f"{_SELECTOR}.get_global_server_args",
                return_value=SimpleNamespace(attention_backend=None),
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform", CpuPlatform
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.cpu.cpu_has_amx_support",
                return_value=True,
            ),
        ):
            backend_cls = get_attn_backend(
                128,
                dtype,
                supported_attention_backends=supported,
                selected_attention_backend=selected,
            )
        return backend_cls.get_enum()

    def test_layer_without_amx_attn_falls_back_to_sdpa(self):
        backend = self._resolve(
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA}
        )
        self.assertEqual(backend, AttentionBackendEnum.TORCH_SDPA)

    def test_explicit_sdpa_is_honored(self):
        backend = self._resolve(
            supported={AttentionBackendEnum.TORCH_SDPA, AttentionBackendEnum.AMX_ATTN},
            selected=AttentionBackendEnum.TORCH_SDPA,
        )
        self.assertEqual(backend, AttentionBackendEnum.TORCH_SDPA)

    def test_fp32_layer_uses_sdpa(self):
        backend = self._resolve(
            supported={AttentionBackendEnum.TORCH_SDPA, AttentionBackendEnum.AMX_ATTN},
            dtype=torch.float32,
        )
        self.assertEqual(backend, AttentionBackendEnum.TORCH_SDPA)

    def test_bf16_layer_that_allows_amx_attn_gets_it(self):
        backend = self._resolve(
            supported={AttentionBackendEnum.TORCH_SDPA, AttentionBackendEnum.AMX_ATTN}
        )
        self.assertEqual(backend, AttentionBackendEnum.AMX_ATTN)


class TestAMXAttentionBatch(CustomTestCase):
    def test_every_batch_is_attended(self):
        """forward used to attend only query[0], so a batch of two came back
        as a batch of one."""
        torch.manual_seed(0)
        batch, seqlen, heads, head_size = 2, 64, 4, 64
        query, key, value = (
            torch.randn(batch, seqlen, heads, head_size, dtype=torch.bfloat16)
            for _ in range(3)
        )
        impl = AMXATTNImpl(
            num_heads=heads,
            head_size=head_size,
            causal=False,
            softmax_scale=head_size**-0.5,
        )

        out = impl.forward(query, key, value, attn_metadata=None)

        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)
        tolerance = precision[torch.bfloat16]
        torch.testing.assert_close(out, expected, atol=tolerance, rtol=tolerance)


class TestQwenImagePackedAddedQKV(CustomTestCase):
    def test_split_reads_packed_weight(self):
        """Qwen-Image's lossless path used to slice to_added_qkv.weight by rows
        and run a plain linear per shard. Once AMX packing has run, the weight
        is VNNI-laid-out under an unchanged [N, K] shape and the bias is fp32,
        so that path failed or returned wrong values."""
        torch.manual_seed(0)
        shard_size, in_features = 128, 256
        weight = torch.randn(3 * shard_size, in_features, dtype=torch.bfloat16) * 0.02
        bias = torch.randn(3 * shard_size, dtype=torch.float32) * 0.02
        x = torch.randn(2, 16, in_features, dtype=torch.bfloat16)
        packed_linear = SimpleNamespace(
            weight=torch.ops.sgl_kernel.convert_weight_packed(weight),
            bias=bias,
            output_partition_sizes=[shard_size] * 3,
            use_intel_amx_backend=True,
        )

        shards = _split_unquantized_merged_linear(packed_linear, x)

        expected = F.linear(x.float(), weight.float(), bias).split(shard_size, dim=-1)
        tolerance = precision[torch.bfloat16]
        for shard, expected_shard in zip(shards, expected):
            torch.testing.assert_close(
                shard.float(), expected_shard, atol=tolerance, rtol=tolerance
            )


class TestDiffusionCpuFusedOps(CustomTestCase):
    def test_fused_residual_gate_add_cpu_matches_eager(self):
        from sglang.kernels.ops.diffusion import residual_gate_add

        torch.manual_seed(0)
        for dtype in (torch.bfloat16, torch.float16):
            for gate_dtype in (dtype, torch.float32):
                for gate_seq in (1, 17):
                    with self.subTest(
                        dtype=dtype, gate_dtype=gate_dtype, gate_seq=gate_seq
                    ):
                        residual = torch.randn(2, 17, 70, dtype=dtype)
                        update = torch.randn(2, 17, 70, dtype=dtype)
                        gate = torch.randn(2, gate_seq, 70, dtype=gate_dtype)
                        actual = residual_gate_add(residual, update, gate)
                        expected = (residual + update * gate).to(dtype)
                        tol = 0.0 if gate_dtype == dtype else precision[dtype]
                        torch.testing.assert_close(
                            actual, expected, atol=tol, rtol=tol
                        )

    def test_apply_interleaved_rope_cpu_matches_sana_video_reference(self):
        from sglang.multimodal_gen.runtime.models.dits.sana_video import (
            apply_interleaved_rotary_emb,
            apply_interleaved_rotary_emb_pair,
        )

        torch.manual_seed(1)
        for q_dtype in (torch.bfloat16, torch.float16, torch.float32):
            for cos_dtype in (torch.float64, torch.float32):
                with self.subTest(q_dtype=q_dtype, cos_dtype=cos_dtype):
                    q = torch.randn(2, 19, 6, 72, dtype=q_dtype)
                    k = torch.randn(2, 19, 6, 72, dtype=q_dtype)
                    base_cos = torch.randn(1, 19, 1, 36, dtype=cos_dtype)
                    base_sin = torch.randn(1, 19, 1, 36, dtype=cos_dtype)
                    cos = base_cos.repeat_interleave(2, dim=-1)
                    sin = base_sin.repeat_interleave(2, dim=-1)

                    q_out, k_out = apply_interleaved_rotary_emb_pair(q, k, cos, sin)
                    q_ref = apply_interleaved_rotary_emb(q, cos, sin)
                    k_ref = apply_interleaved_rotary_emb(k, cos, sin)
                    tol = 0.0 if cos_dtype == torch.float64 else precision[q_dtype]
                    torch.testing.assert_close(q_out, q_ref, atol=tol, rtol=tol)
                    torch.testing.assert_close(k_out, k_ref, atol=tol, rtol=tol)

    def test_anima_and_sana_cpu_fusions_match_eager(self):
        from sglang.multimodal_gen.runtime.layers.layernorm import (
            RMSNorm,
            apply_qk_norm,
        )
        from sglang.multimodal_gen.runtime.models.dits.anima import (
            AnimaAdaLayerNorm,
            _anima_rope,
            _anima_rope_eager,
        )
        from sglang.multimodal_gen.runtime.models.dits.sana import (
            _eager_ln_modulate,
            sana_ln_modulate,
        )

        torch.manual_seed(2)
        q = torch.randn(2, 15, 4, 64, dtype=torch.bfloat16)
        k = torch.randn(2, 15, 4, 64, dtype=torch.bfloat16)
        cos = torch.randn(15, 64, dtype=torch.float32)
        sin = torch.randn(15, 64, dtype=torch.float32)
        q_rot, k_rot = _anima_rope(q, k, cos, sin)
        q_ref, k_ref = _anima_rope_eager(q, k, cos, sin)
        torch.testing.assert_close(
            q_rot, q_ref, atol=precision[torch.bfloat16], rtol=precision[torch.bfloat16]
        )
        torch.testing.assert_close(
            k_rot, k_ref, atol=precision[torch.bfloat16], rtol=precision[torch.bfloat16]
        )

        adaln = AnimaAdaLayerNorm(128, 32).to(torch.bfloat16)
        torch.nn.init.normal_(adaln.linear_1.weight, std=0.02)
        torch.nn.init.normal_(adaln.linear_2.weight, std=0.02)
        x = torch.randn(2, 13, 128, dtype=torch.bfloat16)
        emb_t = torch.randn(2, 128, dtype=torch.bfloat16)
        temb = torch.randn(2, 384, dtype=torch.bfloat16)
        vals = adaln.project_modulation(emb_t, temb)
        out_cpp, gate_cpp = adaln(x, emb_t, temb, values=vals)
        out_ref = adaln.norm(x) * (1 + vals[1]) + vals[0]
        torch.testing.assert_close(
            out_cpp,
            out_ref,
            atol=precision[torch.bfloat16],
            rtol=precision[torch.bfloat16],
        )
        torch.testing.assert_close(gate_cpp, vals[2], atol=0, rtol=0)

        ln = torch.nn.LayerNorm(128, elementwise_affine=False, eps=1e-6)
        x_t = torch.randn(2, 128, 13, dtype=torch.bfloat16).transpose(1, 2)
        scale = torch.randn(2, 1, 128, dtype=torch.bfloat16)
        shift = torch.randn(2, 1, 128, dtype=torch.bfloat16)
        torch.testing.assert_close(
            sana_ln_modulate(ln, x_t, scale, shift, allow_eager=True),
            _eager_ln_modulate(ln, x_t, scale, shift),
            atol=precision[torch.bfloat16],
            rtol=precision[torch.bfloat16],
        )

        q_norm = RMSNorm(64, eps=1e-6).to(torch.bfloat16)
        k_norm = RMSNorm(64, eps=1e-6).to(torch.bfloat16)
        q_in = torch.randn(2, 11, 4, 64, dtype=torch.bfloat16)
        k_in = torch.randn(2, 11, 4, 64, dtype=torch.bfloat16)
        q_expected = q_norm(q_in.clone())
        k_expected = k_norm(k_in.clone())
        q_actual, k_actual = apply_qk_norm(q_in, k_in, q_norm, k_norm, 64)
        torch.testing.assert_close(
            q_actual,
            q_expected,
            atol=precision[torch.bfloat16],
            rtol=precision[torch.bfloat16],
        )
        torch.testing.assert_close(
            k_actual,
            k_expected,
            atol=precision[torch.bfloat16],
            rtol=precision[torch.bfloat16],
        )

    def test_sana_video_linear_attention_cpu_layout_matches_reference(self):
        torch.manual_seed(3)
        b, n, h, d = 2, 65, 4, 32
        query = F.relu(torch.randn(b, n, h, d, dtype=torch.bfloat16))
        key = F.relu(torch.randn(b, n, h, d, dtype=torch.bfloat16))
        query_rotate = torch.randn(b, n, h, d, dtype=torch.bfloat16)
        key_rotate = torch.randn(b, n, h, d, dtype=torch.bfloat16)
        value = torch.randn(b, n, h, d, dtype=torch.bfloat16)

        q_ref = query.permute(0, 2, 3, 1)
        k_ref = key.permute(0, 2, 3, 1)
        qr_ref = query_rotate.permute(0, 2, 3, 1)
        kr_ref = key_rotate.permute(0, 2, 3, 1)
        v_ref = value.permute(0, 2, 3, 1)
        norm_ref = 1.0 / (k_ref.sum(dim=-1, keepdim=True).transpose(-2, -1) @ q_ref + 1e-15)
        scores_ref = v_ref.float() @ kr_ref.float().transpose(-1, -2)
        out_ref = ((scores_ref @ qr_ref.float()) * norm_ref).flatten(1, 2).transpose(1, 2).to(torch.bfloat16)

        q_nhd = query.transpose(1, 2)
        k_sum = key.sum(dim=1).unsqueeze(-1)
        norm_cpu = 1.0 / (q_nhd @ k_sum + 1e-15)
        qr_f = query_rotate.transpose(1, 2).contiguous().float()
        kr_f = key_rotate.transpose(1, 2).contiguous().float()
        v_f = value.transpose(1, 2).contiguous().float()
        scores_cpu = v_f.transpose(-1, -2) @ kr_f
        out_cpu = ((qr_f @ scores_cpu.transpose(-1, -2)) * norm_cpu).transpose(1, 2).reshape(b, n, h * d).to(torch.bfloat16)

        tol = precision[torch.bfloat16]
        torch.testing.assert_close(out_cpu, out_ref, atol=tol, rtol=tol)

    def test_cpu_worker_binds_threads_before_gpu_worker_init(self):
        from sglang.multimodal_gen.runtime.managers import cpu_worker, gpu_worker

        events = []
        with (
            patch.object(cpu_worker, "_is_cpu_amx_available", True),
            patch.object(
                cpu_worker.CPUWorker,
                "init_cpu_threads_binding",
                lambda self: events.append(("bind", self.rank)),
            ),
            patch.object(
                gpu_worker.GPUWorker,
                "__init__",
                lambda self, *args, **kwargs: events.append("super_init"),
            ),
        ):
            cpu_worker.CPUWorker(
                local_rank=0,
                rank=0,
                master_port=30005,
                server_args=SimpleNamespace(num_gpus=1, nnodes=1),
            )
        self.assertEqual(events, [("bind", 0), "super_init"])

    def test_sana_video_amx_pointwise_and_linear_match_eager(self):
        from sglang.multimodal_gen.runtime.models.dits import sana_video

        torch.manual_seed(4)
        ff = sana_video.GLUMBTempConv(64, 2.0).to(torch.bfloat16).eval()
        x = torch.randn(2, 3, 4, 4, 64, dtype=torch.bfloat16)
        with patch.object(sana_video, "_has_cpu_amx", False):
            expected = ff(x)
        actual = ff(x)
        tol = precision[torch.bfloat16]
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


if __name__ == "__main__":
    unittest.main()
