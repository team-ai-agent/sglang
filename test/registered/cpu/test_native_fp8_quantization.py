import unittest

import torch

from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.utils import is_cpu
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu-intel")


@unittest.skipUnless(is_cpu(), "Requires SGLANG_USE_CPU_ENGINE=1")
class TestNativeFp8Quantization(unittest.TestCase):
    def test_static_and_dynamic_quantization_with_strides_and_padding(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            value = torch.tensor(
                [[0, 99, 448, 99, -224, 99, 112, 99], [1, 99, -2, 99, 4, 99, 0, 99]],
                dtype=dtype,
            )[:, ::2]
            original = value.clone()
            self.assertFalse(value.is_contiguous())
            for mode in ("static", "tensor", "token"):
                for padding in (None, 0, 1, 5):
                    with self.subTest(dtype=dtype, mode=mode, padding=padding):
                        static_scale = torch.tensor([2.0]) if mode == "static" else None
                        quantized, scale = scaled_fp8_quant(
                            value,
                            scale=static_scale,
                            num_token_padding=padding,
                            use_per_token_if_dynamic=mode == "token",
                        )
                        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
                        self.assertEqual(quantized.shape, (max(padding or 0, 2), 4))
                        self.assertEqual(scale.dtype, torch.float32)
                        if mode == "static":
                            self.assertIs(scale, static_scale)
                            expected = torch.tensor(
                                [[0, 224, -112, 56], [0.5, -1, 2, 0]]
                            )
                            torch.testing.assert_close(scale, torch.tensor([2.0]))
                        elif mode == "tensor":
                            expected = original.float()
                            torch.testing.assert_close(scale, torch.tensor([1.0]))
                        else:
                            expected = torch.tensor(
                                [[0, 448, -224, 112], [112, -224, 448, 0]],
                                dtype=torch.float32,
                            )
                            self.assertEqual(scale.shape, (quantized.shape[0], 1))
                            torch.testing.assert_close(
                                scale[:2],
                                torch.tensor([[1.0], [4 / 448]]),
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                scale[2:], torch.ones_like(scale[2:]), rtol=0, atol=0
                            )
                        torch.testing.assert_close(
                            quantized[:2].float(), expected, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            quantized[2:].float(),
                            torch.zeros(quantized.shape[0] - 2, 4),
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(value, original, rtol=0, atol=0)

    def test_zero_and_small_values_keep_finite_nonzero_fp32_scales(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for fill in (0.0, 2**-20):
                for per_token in (False, True):
                    with self.subTest(dtype=dtype, fill=fill, per_token=per_token):
                        value = torch.full((2, 4), fill, dtype=dtype)
                        quantized, scale = scaled_fp8_quant(
                            value, use_per_token_if_dynamic=per_token
                        )
                        self.assertTrue(torch.isfinite(quantized.float()).all())
                        self.assertTrue(torch.isfinite(scale).all())
                        self.assertTrue((scale > 0).all())
                        self.assertEqual(scale.dtype, torch.float32)
                        torch.testing.assert_close(
                            quantized.float() * scale, value.float(), atol=0, rtol=1e-6
                        )

    def test_invalid_shapes_are_explicit_errors(self):
        with self.assertRaisesRegex(AssertionError, "Expected 2D"):
            scaled_fp8_quant(torch.ones(4))
        with self.assertRaisesRegex(AssertionError, "Expected scalar scale"):
            scaled_fp8_quant(torch.ones(2, 4), scale=torch.ones(2))


if __name__ == "__main__":
    unittest.main()
