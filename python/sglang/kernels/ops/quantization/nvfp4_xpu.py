import torch
import triton
import triton.language as tl


@triton.jit
def _dequantize_nvfp4(
    packed,
    scales,
    multiplier,
    output,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    PACKED_ROW: tl.constexpr,
    PACKED_COLUMN: tl.constexpr,
    SCALE_ROW: tl.constexpr,
    SCALE_COLUMN: tl.constexpr,
    MULTIPLIER_STRIDE: tl.constexpr,
    ROWS_PER_MULTIPLIER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < ROWS * COLUMNS
    row, column = offset // COLUMNS, offset % COLUMNS
    byte = tl.load(
        packed + row * PACKED_ROW + (column // 2) * PACKED_COLUMN,
        mask=valid,
        other=0,
    ).to(tl.uint32)
    code = (byte >> ((column % 2) * 4)) & 15
    magnitude = (code & 7).to(tl.float32)
    magnitude = tl.where(
        magnitude < 4,
        magnitude * 0.5,
        tl.where(magnitude < 6, magnitude - 2, (magnitude - 4) * 2),
    )
    value = tl.where((code & 8) != 0, -magnitude, magnitude)
    scale_code = tl.load(
        scales + row * SCALE_ROW + (column // 16) * SCALE_COLUMN,
        mask=valid,
        other=0,
    ).to(tl.uint32)
    exponent, mantissa = (scale_code >> 3) & 15, scale_code & 7
    bits = ((exponent + 120) << 23) | (mantissa << 20) | ((scale_code & 128) << 24)
    scale = bits.to(tl.float32, bitcast=True)
    subnormal = mantissa.to(tl.float32) * (1.0 / 512)
    subnormal = tl.where((scale_code & 128) != 0, -subnormal, subnormal)
    scale = tl.where(exponent == 0, subnormal, scale)
    scale = tl.where((exponent == 15) & (mantissa == 7), float("nan"), scale)
    factor = tl.load(
        multiplier + (row // ROWS_PER_MULTIPLIER) * MULTIPLIER_STRIDE,
        mask=valid,
        other=0,
    )
    value = (value * scale) * factor
    tl.store(output + offset, value, mask=valid)


def dequantize_nvfp4(
    *,
    packed: torch.Tensor,
    scales: torch.Tensor,
    global_multiplier: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decode XPU E2M1 uint8[N,K/2], E4M3FN[N,K/16], FP32 multipliers to [N,K].

    Multipliers may be scalar, per equal-width projection, or per output row.
    """
    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError("NVFP4 requires a two-dimensional uint8 packed matrix")
    rows, half_columns = packed.shape
    columns = half_columns * 2
    if rows <= 0 or columns <= 0 or columns % 16:
        raise ValueError("NVFP4 requires nonempty rows and group-16 input columns")
    if scales.shape != (rows, columns // 16) or scales.dtype != torch.float8_e4m3fn:
        raise ValueError("NVFP4 requires one E4M3FN scale per16 input values")
    if (
        packed.device.type != "xpu"
        or scales.device != packed.device
        or global_multiplier.device != packed.device
    ):
        raise ValueError("NVFP4 decode tensors must share an Intel XPU device")
    if global_multiplier.dtype != torch.float32:
        raise ValueError("NVFP4 global multipliers must use FP32")
    count = global_multiplier.numel()
    if count == 0 or rows % count:
        raise ValueError("NVFP4 global multipliers do not match projection rows")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("NVFP4 output must use FP16, BF16 or FP32")
    output = torch.empty((rows, columns), device=packed.device, dtype=dtype)
    multipliers = global_multiplier.reshape(-1)
    _dequantize_nvfp4[(triton.cdiv(rows * columns, 1024),)](
        packed,
        scales.view(torch.uint8),
        multipliers,
        output,
        rows,
        columns,
        *packed.stride(),
        *scales.stride(),
        multipliers.stride(0),
        rows // count,
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
