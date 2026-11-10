import unittest

import sgl_kernel  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading
from sglang.srt.layers.conv import Conv3dLayer
from sglang.srt.utils import cpu_has_amx_support
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import precision

register_cpu_ci(est_time=3, suite="stage-a-test-cpu-intel")


@unittest.skipUnless(cpu_has_amx_support(), "Requires Intel AMX")
class TestConv3dAmxFallback(unittest.TestCase):
    def test_rgb_weights_fall_back_without_changing_dtype(self):
        for dtype in (torch.float32, torch.bfloat16):
            for channels, outputs, kernel in (
                (3, 32, (2, 2, 2)),
                (32, 32, (2, 2, 2)),
                (3, 16, (2, 4, 8)),
            ):
                with self.subTest(dtype=dtype, channels=channels, outputs=outputs):
                    layer = Conv3dLayer(channels, outputs, kernel, stride=kernel).to(dtype)
                    value = torch.randn(2, channels, *(2 * k for k in kernel), dtype=dtype)
                    expected = F.conv3d(value, layer.weight, layer.bias, stride=kernel)
                    _amx_process_weight_after_loading(layer, ["weight"])
                    self.assertFalse(layer.use_intel_amx_backend)
                    actual = layer.forward_cpu(value)
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(
                        actual, expected, atol=precision[dtype], rtol=precision[dtype]
                    )

    def test_aligned_bf16_weights_keep_amx_backend(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                kernel = (2, 4, 8)
                layer = Conv3dLayer(3, 32, kernel, stride=kernel).to(dtype)
                value = torch.randn(2, 3, *kernel, dtype=dtype)
                expected = F.conv3d(value, layer.weight, layer.bias, stride=kernel).flatten(1)
                _amx_process_weight_after_loading(layer, ["weight"])
                self.assertTrue(layer.use_intel_amx_backend)
                torch.testing.assert_close(
                    layer.forward_cpu(value), expected,
                    atol=precision[dtype], rtol=precision[dtype],
                )

    @torch.no_grad()
    def test_depthwise_conv1d_still_uses_its_separate_packing(self):
        layer = nn.Conv1d(32, 32, 4, groups=32, bias=False).bfloat16()
        _amx_process_weight_after_loading(layer, ["weight"])
        self.assertTrue(layer.use_intel_amx_backend)


if __name__ == "__main__":
    unittest.main()
