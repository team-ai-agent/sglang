import unittest

import torch

from sglang.srt.layers.quantization.utils import requantize_with_max_scale
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-test-cpu-intel")
register_cpu_ci(est_time=1, suite="base-b-test-cpu-arm64")


class TestFp8Requantize(CustomTestCase):
    def test_unfused_scales(self):
        torch.manual_seed(42)
        widths = [4, 8, 2]
        weight = (torch.randn(sum(widths), 32) * 40).to(torch.float8_e4m3fn)
        original = weight.clone()
        scales = torch.tensor([0.01, 0.04, 0.02])
        expected = torch.cat(
            [
                (part.float() * scale / scales.max())
                .clamp(-448, 448)
                .to(weight.dtype)
                for part, scale in zip(original.split(widths), scales)
            ]
        )
        scale, result = requantize_with_max_scale(weight, scales, widths)
        torch.testing.assert_close(scale, scales.max())
        torch.testing.assert_close(result.float(), expected.float(), rtol=0, atol=0)
        self.assertEqual(result.data_ptr(), weight.data_ptr())
        self.assertTrue(torch.isfinite(result.float()).all())

    def test_fused_checkpoint_keeps_weights(self):
        weight = torch.tensor([[1.0, -2.0], [4.0, 0.0]]).to(torch.float8_e4m3fn)
        original = weight.clone()
        scales = torch.tensor([0.05, torch.finfo(torch.float8_e4m3fn).min])
        scale, result = requantize_with_max_scale(weight, scales, [1, 1])
        torch.testing.assert_close(scale, scales[0])
        torch.testing.assert_close(result.float(), original.float(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
