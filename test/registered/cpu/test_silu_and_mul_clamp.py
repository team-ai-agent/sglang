import unittest
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.activation.silu_and_mul_clamp import silu_and_mul_clamp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu-intel")


class TestSiluAndMulClamp(unittest.TestCase):
    def test_matches_native_bf16_clamp_order(self):
        for use_cpp_kernel in (True, False):
            generator = torch.Generator().manual_seed(11)
            for dtype in (torch.float16, torch.bfloat16):
                for width in (8, 33):
                    for rows in (0, 3):
                        for limit in (0.0, 6.7, 7.0):
                            for contiguous in (False, True):
                                with self.subTest(
                                    use_cpp=use_cpp_kernel,
                                    dtype=dtype,
                                    width=width,
                                    rows=rows,
                                    limit=limit,
                                    contiguous=contiguous,
                                ):
                                    source = torch.randn(
                                        rows, 4 * width, generator=generator
                                    )
                                    source = (source * 10).to(dtype)
                                    x = (
                                        source[:, ::2].contiguous()
                                        if contiguous
                                        else source[:, ::2]
                                    )
                                    storage = torch.empty(
                                        rows, 2 * width, dtype=dtype
                                    )
                                    output = (
                                        torch.empty(rows, width, dtype=dtype)
                                        if contiguous
                                        else storage[:, ::2]
                                    )
                                    gate, up = x.to(torch.bfloat16).chunk(2, -1)
                                    gate = gate.clamp(max=limit).float()
                                    up = up.clamp(-limit, limit).float()
                                    expected = (gate * gate.sigmoid() * up).to(dtype)
                                    original = x.clone()
                                    ctx = (
                                        patch.object(
                                            torch.ops.sgl_kernel,
                                            "silu_and_mul_clamp_cpu",
                                            new=None,
                                        )
                                        if not use_cpp_kernel
                                        and hasattr(
                                            torch.ops.sgl_kernel,
                                            "silu_and_mul_clamp_cpu",
                                        )
                                        else patch(
                                            "sglang.kernels.ops.activation.silu_and_mul_clamp."
                                            "_jit_silu_and_mul_clamp_module",
                                            side_effect=AssertionError(
                                                "CPU must not invoke CUDA JIT"
                                            ),
                                        )
                                    )
                                    if not use_cpp_kernel and hasattr(
                                        torch.ops.sgl_kernel, "silu_and_mul_clamp_cpu"
                                    ):
                                        saved = torch.ops.sgl_kernel.silu_and_mul_clamp_cpu
                                        delattr(
                                            torch.ops.sgl_kernel,
                                            "silu_and_mul_clamp_cpu",
                                        )
                                        try:
                                            self.assertIsNone(
                                                silu_and_mul_clamp(x, output, limit)
                                            )
                                        finally:
                                            setattr(
                                                torch.ops.sgl_kernel,
                                                "silu_and_mul_clamp_cpu",
                                                saved,
                                            )
                                    else:
                                        with ctx:
                                            self.assertIsNone(
                                                silu_and_mul_clamp(x, output, limit)
                                            )
                                    torch.testing.assert_close(
                                        output,
                                        expected,
                                        rtol=0.001 if dtype == torch.float16 else 0.008,
                                        atol=1e-6,
                                    )
                                    torch.testing.assert_close(
                                        x, original, rtol=0, atol=0
                                    )

    def test_invalid_shape_and_dtype(self):
        with self.assertRaisesRegex(AssertionError, "last dimension"):
            silu_and_mul_clamp(
                torch.ones(2, 16).bfloat16(), torch.empty(2, 7).bfloat16(), 7
            )
        with self.assertRaisesRegex(AssertionError, "FP16/BF16"):
            silu_and_mul_clamp(torch.ones(2, 16), torch.empty(2, 8), 7)

    def test_cuda_dispatch_is_unchanged(self):
        x = Mock(spec=torch.Tensor)
        x.device = torch.device("cuda")
        x.dtype = torch.bfloat16
        output = Mock(spec=torch.Tensor)
        module = Mock()
        with patch(
            "sglang.kernels.ops.activation.silu_and_mul_clamp."
            "_jit_silu_and_mul_clamp_module",
            return_value=module,
        ) as jit:
            silu_and_mul_clamp(x, output, 7)
            jit.assert_called_once_with(x.dtype)
            module.run.assert_called_once_with(x, output, 7.0)


if __name__ == "__main__":
    unittest.main()
