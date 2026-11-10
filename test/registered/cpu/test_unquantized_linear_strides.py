import unittest

import sgl_kernel  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import cpu_has_amx_support
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import precision

register_cpu_ci(est_time=3, suite="stage-a-test-cpu-intel")


@unittest.skipUnless(cpu_has_amx_support(), "Requires Intel AMX")
class TestUnquantizedLinearStrides(unittest.TestCase):
    def test_strided_and_contiguous_activations(self):
        torch.manual_seed(7)
        for has_bias in (False, True):
            layer = nn.Linear(64, 32, bias=has_bias, dtype=torch.bfloat16)
            weight = layer.weight.detach().clone()
            bias = layer.bias.detach().clone() if has_bias else None
            method = UnquantizedLinearMethod()
            method.process_weights_after_loading(layer)
            self.assertTrue(layer.use_intel_amx_backend)
            inputs = [
                torch.randn(7, 64, dtype=torch.bfloat16),
                torch.randn(2, 7, 64, dtype=torch.bfloat16),
                torch.randn(7, 128, dtype=torch.bfloat16)[:, ::2],
                torch.randn(2, 7, 128, dtype=torch.bfloat16)[:, :, ::2],
                torch.randn(7, 2, 64, dtype=torch.bfloat16).transpose(0, 1),
            ]
            for index, value in enumerate(inputs):
                with self.subTest(has_bias=has_bias, layout=index):
                    expected = F.linear(value, weight, bias)
                    actual = method.apply(layer, value, layer.bias)
                    self.assertEqual(actual.dtype, value.dtype)
                    torch.testing.assert_close(
                        actual, expected,
                        atol=precision[value.dtype], rtol=precision[value.dtype],
                    )


if __name__ == "__main__":
    unittest.main()
