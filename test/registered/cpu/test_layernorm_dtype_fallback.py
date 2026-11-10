import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from sglang.srt.layers.layernorm import LayerNorm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


class TestLayerNormCpuDtype(unittest.TestCase):
    @torch.inference_mode()
    def test_mixed_precision_uses_native_accumulation(self):
        torch.manual_seed(0)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                layer = LayerNorm(32, dtype=torch.float32)
                x = torch.randn(2, 3, 32, dtype=dtype)
                expected = F.layer_norm(
                    x.float(), (32,), layer.weight, layer.bias, layer.variance_epsilon
                ).to(dtype)
                actual = layer.forward_cpu(x)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, expected)

    @torch.inference_mode()
    def test_fp32_and_non_affine_use_native(self):
        for dtype, affine, bias in [
            (torch.float32, True, True),
            (torch.bfloat16, False, False),
        ]:
            with self.subTest(dtype=dtype, affine=affine):
                layer = LayerNorm(32, dtype=dtype, elementwise_affine=affine, bias=bias)
                x = torch.randn(2, 32, dtype=dtype)
                with mock.patch.object(
                    layer, "forward_native", wraps=layer.forward_native
                ) as native:
                    torch.testing.assert_close(
                        layer.forward_cpu(x), layer.forward_native(x)
                    )
                    self.assertEqual(native.call_count, 2)

    @torch.inference_mode()
    def test_matching_reduced_precision_keeps_cpu_kernel(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                layer = LayerNorm(32, dtype=dtype)
                x = torch.randn(2, 3, 32, dtype=dtype)
                with mock.patch.object(
                    torch.ops.sgl_kernel,
                    "layernorm_cpu",
                    wraps=torch.ops.sgl_kernel.layernorm_cpu,
                ) as kernel:
                    actual = layer.forward_cpu(x)
                    kernel.assert_called_once()
                torch.testing.assert_close(
                    actual,
                    F.layer_norm(
                        x, (32,), layer.weight, layer.bias, layer.variance_epsilon
                    ),
                    atol=0.02,
                    rtol=0.02,
                )


if __name__ == "__main__":
    unittest.main()
