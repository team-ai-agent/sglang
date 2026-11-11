import unittest

import torch

from sglang.srt.models.minicpmv import (
    MiniCPMV,
    MiniCPMV4_6ForConditionalGeneration,
    MiniCPMV4_7ForConditionalGeneration,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


class TestMiniCPMVModelContext(CustomTestCase):
    def test_runner_layout_survives_eval_unwrap(self):
        for cls in (
            MiniCPMV,
            MiniCPMV4_6ForConditionalGeneration,
            MiniCPMV4_7ForConditionalGeneration,
        ):
            with self.subTest(model=cls.__name__):
                adapter = cls.__new__(cls)
                adapter.minicpmv = torch.nn.Linear(2, 2, bias=False)
                layout = object()
                adapter.shared_experts_config = layout

                model = adapter.eval()

                self.assertIs(model, adapter.minicpmv)
                self.assertFalse(model.training)
                self.assertIs(model.shared_experts_config, layout)
                self.assertIs(adapter.shared_experts_config, layout)

    def test_replacing_layout_updates_inner_model(self):
        adapter = MiniCPMV.__new__(MiniCPMV)
        adapter.minicpmv = torch.nn.Linear(2, 2, bias=False)
        adapter.shared_experts_config = object()
        replacement = object()

        adapter.shared_experts_config = replacement

        self.assertIs(adapter.minicpmv.shared_experts_config, replacement)
        self.assertNotIn("shared_experts_config", vars(adapter))


if __name__ == "__main__":
    unittest.main()
