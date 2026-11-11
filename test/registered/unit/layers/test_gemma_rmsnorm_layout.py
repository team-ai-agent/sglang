import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.layernorm import Gemma4RMSNorm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGemmaRMSNormLayout(CustomTestCase):
    def test_audio_convolution_layout_reaches_xpu_kernel_contiguously(self):
        """Conformer channel/time transposes must satisfy the XPU norm kernel ABI."""
        for shift in (0.0, 1.0):
            with self.subTest(scale_shift=shift):
                norm = Gemma4RMSNorm(128, scale_shift=shift)
                x = torch.arange(2 * 128 * 7).reshape(2, 128, 7).float().transpose(1, 2)
                expected = norm.forward_native(x)

                def kernel(tensor, weight, epsilon):
                    self.assertEqual(tensor.stride(-1), 1)
                    return norm.forward_native(tensor)

                with (
                    patch("sglang.srt.layers.layernorm.rmsnorm", side_effect=kernel),
                    patch(
                        "sglang.srt.layers.layernorm.gemma_rmsnorm", side_effect=kernel
                    ),
                ):
                    output = norm.forward_xpu(x)
                torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
                self.assertEqual(x.stride(-1), 7)


if __name__ == "__main__":
    unittest.main()
