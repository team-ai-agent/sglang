import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.interns1 import InternS1ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, forward_batch, input_embeds, positions):
        return input_embeds


class TestInternS1Media(CustomTestCase):
    def test_video_frame_features_reach_language_model_in_order(self):
        """Video items must use the vision encoder and preserve separate frame spans."""
        for modality in (Modality.IMAGE, Modality.VIDEO):
            with self.subTest(modality=modality):
                model = InternS1ForConditionalGeneration.__new__(
                    InternS1ForConditionalGeneration
                )
                nn.Module.__init__(model)
                model.language_model = _LanguageModel()
                model.extract_feature = lambda pixels: pixels
                features = torch.tensor(
                    [[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]]
                )
                item = MultimodalDataItem(
                    modality=modality,
                    hash=17,
                    offsets=[(1, 1), (3, 3)],
                    feature=features,
                )
                item.set_pad_value()
                marker = item.padding_values()[0]
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    contains_mm_inputs=lambda: True,
                    mm_inputs=[MultimodalInputs(mm_items=[item])],
                    extend_prefix_lens_cpu=[0],
                    extend_seq_lens_cpu=[5],
                    spec_algorithm=None,
                    input_embeds=None,
                )
                with (
                    patch.object(mm_schedule, "embedding_cache", None),
                    patch(
                        "sglang.srt.managers.mm_utils.get_server_args",
                        return_value=None,
                    ),
                ):
                    mm_schedule.init_mm_embedding_cache(0)
                    output = model(
                        input_ids=torch.tensor([0, marker, 1, marker, 2]),
                        positions=torch.arange(5),
                        forward_batch=batch,
                    )
                torch.testing.assert_close(output[[1, 3]], features.reshape(2, 4))
                torch.testing.assert_close(
                    output[[0, 2, 4]],
                    model.language_model.embedding(torch.tensor([0, 1, 2])),
                )


if __name__ == "__main__":
    unittest.main()
