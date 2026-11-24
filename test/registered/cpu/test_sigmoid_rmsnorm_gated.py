import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu-intel")


class TestSigmoidRMSNormGated(unittest.TestCase):
    def test_sigmoid_matches_float32_reference(self):
        for use_cpp in (True, False):
            generator = torch.Generator().manual_seed(7)
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for affine in (False, True):
                    for width in (64, 70):
                        with self.subTest(
                            use_cpp=use_cpp, dtype=dtype, affine=affine, width=width
                        ):
                            x = torch.randn(
                                3, 2, 2 * width, generator=generator
                            ).to(dtype)[..., ::2]
                            gate = torch.randn(
                                3, 2, 2 * width, generator=generator
                            ).to(dtype)[..., ::2]
                            layer = FusedRMSNormGated(
                                width, activation="sigmoid", elementwise_affine=affine
                            )
                            if affine:
                                with torch.no_grad():
                                    layer.weight.copy_(
                                        torch.randn(width, generator=generator)
                                    )
                            weight = layer.weight.float() if affine else None
                            expected = (
                                F.rms_norm(x.float(), (width,), weight, layer.eps)
                                * gate.float().sigmoid()
                            ).to(dtype)
                            has_op = hasattr(
                                torch.ops.sgl_kernel, "fused_rmsnorm_sigmoid_gated_cpu"
                            )
                            if not use_cpp and has_op:
                                saved = (
                                    torch.ops.sgl_kernel.fused_rmsnorm_sigmoid_gated_cpu
                                )
                                delattr(
                                    torch.ops.sgl_kernel,
                                    "fused_rmsnorm_sigmoid_gated_cpu",
                                )
                                try:
                                    actual = layer(x, gate.reshape(6, width))
                                finally:
                                    setattr(
                                        torch.ops.sgl_kernel,
                                        "fused_rmsnorm_sigmoid_gated_cpu",
                                        saved,
                                    )
                            else:
                                actual = layer(x, gate.reshape(6, width))
                            self.assertEqual(actual.dtype, dtype)
                            self.assertEqual(actual.shape, x.shape)
                            torch.testing.assert_close(
                                actual,
                                expected,
                                rtol=1e-5 if dtype == torch.float32 else 0.008,
                                atol=1e-6,
                            )

    def test_residual_prenorm_and_dtype(self):
        x = torch.arange(128).reshape(2, 64).to(torch.bfloat16) / 64
        gate = torch.ones_like(x)
        layer = FusedRMSNormGated(64, activation="sigmoid", elementwise_affine=False)
        for residual in (None, torch.ones_like(x), torch.ones_like(x).float()):
            for in_fp32 in (False, True):
                with self.subTest(
                    residual_dtype=getattr(residual, "dtype", None), in_fp32=in_fp32
                ):
                    values = x.float() + (
                        residual.float() if residual is not None else 0
                    )
                    expected = (
                        F.rms_norm(values, (64,), None, layer.eps)
                        * gate.float().sigmoid()
                    ).to(x.dtype)
                    actual, residual_out = layer(
                        x,
                        gate,
                        residual=residual,
                        prenorm=True,
                        residual_in_fp32=in_fp32,
                    )
                    expected_dtype = (
                        residual.dtype
                        if residual is not None
                        else torch.float32
                        if in_fp32
                        else x.dtype
                    )
                    self.assertEqual(residual_out.dtype, expected_dtype)
                    torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-6)
                    torch.testing.assert_close(residual_out, values.to(expected_dtype))

    def test_silu_native_path_is_preserved(self):
        x = torch.ones(2, 64, dtype=torch.bfloat16)
        layer = FusedRMSNormGated(64, activation="silu")
        expected = torch.empty_like(x)
        with patch.object(
            torch.ops.sgl_kernel, "fused_rmsnorm_gated_cpu", return_value=expected
        ) as native:
            self.assertIs(layer(x, x), expected)
            native.assert_called_once_with(x, layer.weight, x, layer.eps)

    def test_invalid_shapes(self):
        layer = FusedRMSNormGated(64, activation="sigmoid")
        x = torch.ones(2, 64)
        with self.assertRaisesRegex(AssertionError, "Gate shape"):
            layer(x, torch.ones(2, 32))
        with self.assertRaisesRegex(AssertionError, "Residual shape"):
            layer(x, x, residual=torch.ones(1, 64))


if __name__ == "__main__":
    unittest.main()
