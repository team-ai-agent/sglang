import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.internlm_detector import InternlmDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestInternlmDetector(CustomTestCase):
    def test_native_and_legacy_marker_spacing(self):
        """Adjacent native Intern-S1 action tokens must parse like legacy tokens."""
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                ),
            )
        ]
        for gap in ("", " "):
            text = (
                "<|action_start|>" + gap + "<|plugin|>\n"
                '{"name":"get_weather","parameters":{"city":"Beijing"}}'
                "<|action_end|>"
            )
            with self.subTest(gap=gap, streaming=False):
                parser = InternlmDetector()
                self.assertTrue(parser.has_tool_call(text))
                result = parser.detect_and_parse(text=text, tools=tools)
                self.assertEqual(result.normal_text, "")
                self.assertEqual(result.calls[0].name, "get_weather")
                self.assertEqual(
                    json.loads(result.calls[0].parameters), {"city": "Beijing"}
                )
            with self.subTest(gap=gap, streaming=True):
                parser = InternlmDetector()
                parts = [
                    parser.parse_streaming_increment(new_text=char, tools=tools)
                    for char in text
                ]
                calls = [call for part in parts for call in part.calls]
                self.assertEqual("".join(part.normal_text for part in parts), "")
                self.assertEqual(calls[0].name, "get_weather")
                self.assertEqual(json.loads(calls[0].parameters), {"city": "Beijing"})


if __name__ == "__main__":
    unittest.main()
