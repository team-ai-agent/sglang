import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from transformers.models.gemma4_unified.processing_gemma4_unified import (
    Gemma4UnifiedProcessor,
)

from sglang.srt.managers.multimodal_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.gemma4 import Gemma4SGLangProcessor
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestGemma4UnifiedVideoMetadata(CustomTestCase):
    def test_native_prompt_preserves_source_timestamps_after_sampling(self):
        """Decoded frames must retain source timing instead of HF's fallback24FPS."""
        decoder = MagicMock(spec=VideoDecoderWrapper)
        decoder.__len__.return_value = 16
        decoder.avg_fps = 4.0
        decoder.get_frames_at.side_effect = lambda indices: np.stack(
            [np.full((2, 2, 3), index, dtype=np.uint8) for index in indices]
        )
        processor = object.__new__(Gemma4SGLangProcessor)
        processor._processor = SimpleNamespace(
            video_processor=SimpleNamespace(num_frames=4)
        )
        with patch.object(BaseMultimodalProcessor, "process_mm_data") as native_call:
            processor.process_mm_data(input_text="<video>", videos=[decoder])
        forwarded = native_call.call_args.kwargs
        self.assertEqual(forwarded["videos"][0][:, 0, 0, 0].tolist(), [0, 4, 8, 12])
        metadata = forwarded["video_metadata"][0]
        self.assertEqual(metadata.timestamps, [0.0, 1.0, 2.0, 3.0])
        native = SimpleNamespace(
            boi_token="<image>", video_token="<frame>", eoi_token="</image>"
        )
        prompt = Gemma4UnifiedProcessor.replace_video_token(
            native,
            {"num_soft_tokens_per_video": [1], "video_metadata": [metadata]},
            video_idx=0,
        )
        self.assertIn("00:02 <image>", prompt)
        self.assertIn("00:03 <image>", prompt)


if __name__ == "__main__":
    unittest.main()
