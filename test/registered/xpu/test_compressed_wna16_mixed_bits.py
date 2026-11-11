import unittest

import torch
import triton.language as tl
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16TritonMoE,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")


@unittest.skipUnless(torch.xpu.is_available(), "requires Intel XPU")
class TestCompressedWNA16MixedBits(CustomTestCase):
    def test_packed_groupwise_projections_match_dequantized_reference(self):
        """INT8 must keep all eight bits and its offset128 across grouped expert routing."""
        torch.manual_seed(7)
        device = "xpu"
        experts, width, hidden = 2, 256, 256
        activation = torch.randn(2, hidden, device=device, dtype=torch.bfloat16) * 0.1
        for bits in (4, 8):
            for group in (32, 128):
                with self.subTest(bits=bits, group=group):
                    integers = torch.randint(
                        0,
                        1 << bits,
                        (experts, width, hidden),
                        device=device,
                        dtype=torch.uint8,
                    )
                    signed = (integers.to(torch.int16) - (1 << (bits - 1))).to(
                        torch.int8
                    )
                    packed_words = pack_to_int32(signed, num_bits=bits)
                    scales = (
                        torch.rand(
                            experts,
                            width,
                            hidden // group,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        * 0.01
                    )
                    layer = torch.nn.Module()
                    layer.w13_weight_packed = torch.nn.Parameter(
                        packed_words.transpose(1, 2).contiguous(),
                        requires_grad=False,
                    )
                    layer.w2_weight_packed = torch.nn.Parameter(
                        layer.w13_weight_packed.detach().clone(), requires_grad=False
                    )
                    layer.w13_weight_scale = torch.nn.Parameter(
                        scales.transpose(1, 2).contiguous(), requires_grad=False
                    )
                    layer.w2_weight_scale = torch.nn.Parameter(
                        layer.w13_weight_scale.detach().clone(), requires_grad=False
                    )
                    scheme = CompressedTensorsWNA16TritonMoE.__new__(
                        CompressedTensorsWNA16TritonMoE
                    )
                    scheme.num_bits, scheme.group_size = bits, group
                    scheme.process_weights_after_loading(layer)
                    info = scheme.get_triton_quant_info(layer)
                    output = torch.empty(
                        2, 1, width, device=device, dtype=torch.bfloat16
                    )
                    routes = torch.tensor([[0], [1]], device=device, dtype=torch.int32)
                    weights = torch.tensor([[0.3], [0.7]], device=device)
                    invoke_fused_moe_kernel(
                        A=activation,
                        B=info.w13_weight,
                        bias=None,
                        C=output,
                        A_scale=None,
                        B_scale=info.w13_scale,
                        B_zp=None,
                        topk_weights=weights,
                        topk_ids=routes,
                        sorted_token_ids=torch.tensor(
                            [0] + [2] * 15 + [1] + [2] * 15,
                            device=device,
                            dtype=torch.int32,
                        ),
                        expert_ids=torch.tensor(
                            [0, 1], device=device, dtype=torch.int32
                        ),
                        num_tokens_post_padded=torch.tensor(
                            [32], device=device, dtype=torch.int32
                        ),
                        mul_routed_weight=True,
                        top_k=1,
                        config={
                            "BLOCK_SIZE_M": 16,
                            "BLOCK_SIZE_N": 32,
                            "BLOCK_SIZE_K": 64,
                            "GROUP_SIZE_M": 1,
                            "num_warps": 4,
                            "num_stages": 2,
                        },
                        compute_type=tl.bfloat16,
                        use_fp8_w8a8=False,
                        use_int8_w8a8=False,
                        use_int8_w8a16=info.use_int8_w8a16,
                        use_int4_w4a16=info.use_int4_w4a16,
                        per_channel_quant=False,
                        block_shape=info.block_shape,
                    )
                    dequantized = (
                        (
                            (integers.float() - (1 << (bits - 1)))
                            * scales.repeat_interleave(group, dim=-1).float()
                        )
                        .to(torch.bfloat16)
                        .float()
                    )
                    expected = (
                        torch.einsum("mk,mnk->mn", activation.float(), dequantized)
                        * weights
                    )
                    torch.testing.assert_close(
                        output[:, 0].float(), expected, rtol=2e-2, atol=2e-3
                    )


if __name__ == "__main__":
    unittest.main()
