import unittest

import torch

from sglang.kernels.ops.quantization.nvfp4_xpu import dequantize_nvfp4
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")


@unittest.skipUnless(torch.xpu.is_available(), "requires Intel XPU")
class TestNVFP4XPU(CustomTestCase):
    def test_all_e2m1_and_e4m3fn_codes_match_independent_cpu_reference(self):
        """Bit decoding must retain FP8 subnormals, maximums, signs and NaNs."""
        codes = torch.arange(16).expand(256, -1)
        table = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        scales = (
            torch.arange(256, dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .reshape(-1, 1)
        )
        multiplier = torch.tensor([0.1234567], dtype=torch.float32)
        reference = (table[codes] * scales.float()) * multiplier
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                result = dequantize_nvfp4(
                    packed=packed.to("xpu"),
                    scales=scales.to("xpu"),
                    global_multiplier=multiplier.to("xpu"),
                    dtype=dtype,
                ).cpu()
                expected = reference.to(dtype)
                torch.testing.assert_close(
                    result, expected, rtol=0, atol=0, equal_nan=True
                )
                zeros = expected == 0
                self.assertTrue(
                    torch.equal(
                        torch.signbit(result[zeros]), torch.signbit(expected[zeros])
                    )
                )

    def test_strided_tiles_keep_per_row_multipliers_and_tail_mask(self):
        """Output-row tiling must not mix separate fused globals or read padding."""
        raw = torch.arange(37 * 64).reshape(37, 64).to(torch.uint8)
        packed = raw[:, ::2]
        scale_bytes = ((torch.arange(37 * 8).reshape(37, 8) % 80) + 32).to(torch.uint8)
        scales = scale_bytes.view(torch.float8_e4m3fn)[:, ::2]
        storage = torch.linspace(0.000123, 0.9, 74)
        globals_ = storage[::2]
        table = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        reference = torch.empty(37, 64)
        reference[:, ::2] = table[(packed & 15).long()]
        reference[:, 1::2] = table[(packed >> 4).long()]
        reference *= scales.float().repeat_interleave(16, dim=-1)
        reference *= globals_[:, None]
        result = dequantize_nvfp4(
            packed=raw.to("xpu")[:, ::2],
            scales=scale_bytes.to("xpu").view(torch.float8_e4m3fn)[:, ::2],
            global_multiplier=storage.to("xpu")[::2],
            dtype=torch.bfloat16,
        ).cpu()
        torch.testing.assert_close(result, reference.to(torch.bfloat16), rtol=0, atol=0)

    def test_rejects_different_fp8_encodings_and_invalid_global_count(self):
        packed = torch.zeros(3, 8, dtype=torch.uint8, device="xpu")
        for dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "E4M3FN"):
                    dequantize_nvfp4(
                        packed=packed,
                        scales=torch.zeros(3, 1, dtype=dtype, device="xpu"),
                        global_multiplier=torch.ones((), device="xpu"),
                        dtype=torch.bfloat16,
                    )
        with self.assertRaisesRegex(ValueError, "projection rows"):
            dequantize_nvfp4(
                packed=packed,
                scales=torch.zeros(3, 1, dtype=torch.float8_e4m3fn, device="xpu"),
                global_multiplier=torch.ones(2, device="xpu"),
                dtype=torch.bfloat16,
            )


if __name__ == "__main__":
    unittest.main()
