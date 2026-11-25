from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers import vocab_parallel_embedding
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.models import qwen3_moe
from sglang.srt.runtime_context import get_context


@pytest.mark.parametrize(
    "cpu,config_fp32,server_fp32,expected_dtype",
    [
        (True, True, False, torch.float32),
        (True, False, True, torch.float32),
        (True, False, False, torch.bfloat16),
        (False, True, False, torch.bfloat16),
        (False, False, True, torch.bfloat16),
    ],
)
def test_fp32_lm_head_keeps_logical_cpu_weight_layout(
    monkeypatch, cpu, config_fp32, server_fp32, expected_dtype
):
    config = SimpleNamespace(
        vocab_size=128, hidden_size=64, enable_lm_head_fp32=config_fp32
    )
    monkeypatch.setattr(qwen3_moe, "_is_cpu", cpu)
    monkeypatch.setattr(
        qwen3_moe,
        "get_exec",
        lambda: SimpleNamespace(
            features=SimpleNamespace(enable_fp32_lm_head=server_fp32)
        ),
    )
    monkeypatch.setattr(
        qwen3_moe.Qwen3MoeForCausalLM,
        "model_cls",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        qwen3_moe, "LogitsProcessor", lambda config: torch.nn.Identity()
    )
    monkeypatch.setattr(
        vocab_parallel_embedding, "_resolve_linear_group", lambda group: None
    )
    with (
        get_context().parallel.override(
            tp_rank=0,
            tp_size=1,
            attn_cp_size=1,
            moe_dp_size=1,
            pp_group=None,
            enable_dp_lm_head=False,
        ),
        torch.device("cpu"),
    ):
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            model = qwen3_moe.Qwen3MoeForCausalLM(config)
        finally:
            torch.set_default_dtype(previous_dtype)
    assert model.lm_head.weight.dtype == expected_dtype
    if expected_dtype != torch.float32:
        return

    checkpoint = torch.randn(128, 64, dtype=torch.bfloat16)
    model.lm_head.weight.data.copy_(checkpoint)
    model.lm_head.quant_method.process_weights_after_loading(model.lm_head)
    torch.testing.assert_close(model.lm_head.weight, checkpoint.float(), rtol=0, atol=0)

    hidden = torch.randn(3, 64, dtype=torch.bfloat16)
    processor = object.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.use_fp32_lm_head = True
    actual = processor._compute_lm_head(hidden, model.lm_head)
    expected = torch.nn.functional.linear(hidden.float(), checkpoint.float())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
