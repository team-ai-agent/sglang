#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t, typename func_t, typename vec_func_t>
void act_and_mul_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int64_t num_tokens,
    int64_t dim,
    const func_t& f,
    const vec_func_t& vf) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int64_t kVecSize = bVec::size();
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      // local ptrs
      const scalar_t* __restrict__ input_ptr = input + i * 2 * dim;
      const scalar_t* __restrict__ input_other_ptr = input_ptr + dim;
      scalar_t* __restrict__ output_ptr = output + i * dim;

      int64_t d;
#pragma GCC unroll 4
      for (d = 0; d <= dim - kVecSize; d += kVecSize) {
        auto [x_fvec0, x_fvec1] = load_float_vec2(input_ptr + d);
        auto [y_fvec0, y_fvec1] = load_float_vec2(input_other_ptr + d);

        x_fvec0 = vf(x_fvec0);
        x_fvec1 = vf(x_fvec1);

        x_fvec0 = x_fvec0 * y_fvec0;
        x_fvec1 = x_fvec1 * y_fvec1;

        convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1).store(output_ptr + d);
      }
#pragma GCC unroll 4
      for (; d < dim; ++d) {
        float x_val = static_cast<float>(input_ptr[d]);
        float y_val = static_cast<float>(input_other_ptr[d]);
        output_ptr[d] = f(x_val) * y_val;
      }
    }
  });
}

// input : [num_tokens, dim] contiguous
// gate : [num_tokens, num_heads, head_dim] 2d or 3d, maybe strided
template <typename scalar_t>
void fused_sigmoid_mul_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gate,
    int64_t num_tokens,
    int64_t dim,
    int64_t num_heads,
    int64_t head_dim,
    int64_t g_strideT,
    int64_t g_strideH) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int64_t kVecSize = bVec::size();
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const scalar_t* __restrict__ i_ptr = input + i * dim;
      const scalar_t* __restrict__ g_ptr = gate + i * g_strideT;
      scalar_t* __restrict__ o_ptr = output + i * dim;

      for (int64_t h = 0; h < num_heads; ++h) {
        const scalar_t* __restrict__ attn_ptr = i_ptr + h * head_dim;
        const scalar_t* __restrict__ gate_ptr = g_ptr + h * g_strideH;
        scalar_t* __restrict__ out_ptr = o_ptr + h * head_dim;

        int64_t d = 0;
#pragma GCC unroll 4
        for (; d <= head_dim - kVecSize; d += kVecSize) {
          auto [x_fvec0, x_fvec1] = load_float_vec2(attn_ptr + d);
          auto [g_fvec0, g_fvec1] = load_float_vec2(gate_ptr + d);
          x_fvec0 = x_fvec0 * fast_sigmoid(g_fvec0);
          x_fvec1 = x_fvec1 * fast_sigmoid(g_fvec1);
          convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1).store(out_ptr + d);
        }
#pragma GCC unroll 4
        for (; d < head_dim; ++d) {
          float x_val = static_cast<float>(attn_ptr[d]);
          float g_val = static_cast<float>(gate_ptr[d]);
          out_ptr[d] = static_cast<scalar_t>(x_val / (1.f + std::exp(-g_val)));
        }
      }
    }
  });
}

}  // anonymous namespace

// input   : {num_tokens, 2 * d}
// output  : {num_tokens, d}
at::Tensor silu_and_mul_cpu(at::Tensor& input) {
  auto sizes = input.sizes().vec();
  int64_t last_dim = input.ndimension() - 1;
  int64_t d = sizes[last_dim] / 2;
  sizes[last_dim] = d;
  int64_t num_tokens = input.numel() / input.size(-1);
  at::Tensor out = at::empty(sizes, input.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "silu_and_mul", [&] {
    using Vec = at::vec::Vectorized<float>;
    act_and_mul_kernel_impl(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        num_tokens,
        d,
        [](float x) { return x / (1.f + std::exp(-x)); },
        [](Vec x) { return fast_silu(x); });
  });
  return out;
}

at::Tensor gelu_tanh_and_mul_cpu(const at::Tensor& input) {
  auto sizes = input.sizes().vec();
  int64_t last_dim = input.ndimension() - 1;
  int64_t d = sizes[last_dim] / 2;
  sizes[last_dim] = d;
  int64_t num_tokens = input.numel() / input.size(-1);
  at::Tensor out = at::empty(sizes, input.options());
  const float sqrt_2_div_pi = std::sqrt(2.f / M_PI);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "gelu_tanh_and_mul", [&] {
    using Vec = at::vec::Vectorized<float>;
    act_and_mul_kernel_impl(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        num_tokens,
        d,
        [sqrt_2_div_pi](float x) {
          float x3 = x * x * x;
          float tanh_arg = sqrt_2_div_pi * (x + 0.044715f * x3);
          return 0.5f * x * (1.f + std::tanh(tanh_arg));
        },
        [sqrt_2_div_pi](Vec x) {
          Vec x3 = x * x * x;
          Vec tanh_arg = Vec(sqrt_2_div_pi) * (x + Vec(0.044715f) * x3);
          return Vec(0.5f) * x * (Vec(1.f) + tanh_arg.tanh());
        });
  });

  return out;
}

at::Tensor gelu_and_mul_cpu(const at::Tensor& input) {
  auto sizes = input.sizes().vec();
  int64_t last_dim = input.ndimension() - 1;
  int64_t d = sizes[last_dim] / 2;
  sizes[last_dim] = d;
  int64_t num_tokens = input.numel() / input.size(-1);
  at::Tensor out = at::empty(sizes, input.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "gelu_and_mul", [&] {
    using Vec = at::vec::Vectorized<float>;
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    act_and_mul_kernel_impl(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        num_tokens,
        d,
        [inv_sqrt2](float x) { return 0.5f * x * (1.f + std::erf(x * inv_sqrt2)); },
        [](Vec x) { return fast_gelu(x); });
  });

  return out;
}

void fused_sigmoid_mul_cpu(at::Tensor& input, const at::Tensor& gate) {
  CHECK_DIM(2, input);
  const int64_t gate_dim = gate.dim();
  TORCH_CHECK(gate_dim == 2 || gate_dim == 3, "gate must be a 2D or 3D tensor");
  CHECK_CONTIGUOUS(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(gate);

  const auto st = input.scalar_type();
  CHECK_EQ(gate.scalar_type(), st);

  int64_t num_tokens = input.size(0);
  int64_t d = input.size(1);

  const bool is_gate_3d = gate_dim == 3;
  int64_t num_heads = is_gate_3d ? gate.size(1) : 1;
  int64_t head_dim = gate.size(-1);
  CHECK_EQ(gate.size(0), num_tokens);
  CHECK_EQ(d, num_heads * head_dim);

  int64_t g_strideT = gate.stride(0);
  int64_t g_strideH = is_gate_3d ? gate.stride(1) : 0;

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_sigmoid_mul", [&] {
    fused_sigmoid_mul_kernel_impl<scalar_t>(
        input.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        gate.data_ptr<scalar_t>(),
        num_tokens,
        d,
        num_heads,
        head_dim,
        g_strideT,
        g_strideH);
  });
}

namespace {

template <typename scalar_t>
void silu_and_mul_clamp_kernel_impl(
    scalar_t* __restrict__ out_base,
    const scalar_t* __restrict__ in_base,
    int64_t num_tokens,
    int64_t dim,
    int64_t in_s0,
    int64_t in_s1,
    int64_t out_s0,
    int64_t out_s1,
    float limit_lo,
    float limit_hi) {
  if (in_s1 == 1 && out_s1 == 1) {
    using bVec = at::vec::Vectorized<scalar_t>;
    using bfVec = at::vec::Vectorized<at::BFloat16>;
    using fVec = at::vec::Vectorized<float>;
    constexpr int64_t kVecSize = bVec::size();
    const fVec v_hi(limit_hi);
    const fVec v_lo(limit_lo);
    const fVec v_one(1.0f);

    at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const scalar_t* __restrict__ g_ptr = in_base + i * in_s0;
        const scalar_t* __restrict__ u_ptr = g_ptr + dim;
        scalar_t* __restrict__ o_ptr = out_base + i * out_s0;

        int64_t d = 0;
#pragma GCC unroll 4
        for (; d <= dim - kVecSize; d += kVecSize) {
          auto [g0, g1] = load_float_vec2(g_ptr + d);
          auto [u0, u1] = load_float_vec2(u_ptr + d);
          if constexpr (!std::is_same_v<scalar_t, at::BFloat16>) {
            bfVec gb = convert_from_float_ext<at::BFloat16>(g0, g1);
            bfVec ub = convert_from_float_ext<at::BFloat16>(u0, u1);
            std::tie(g0, g1) = at::vec::convert_to_float(gb);
            std::tie(u0, u1) = at::vec::convert_to_float(ub);
          }
          g0 = at::vec::minimum(g0, v_hi);
          g1 = at::vec::minimum(g1, v_hi);
          u0 = at::vec::clamp(u0, v_lo, v_hi);
          u1 = at::vec::clamp(u1, v_lo, v_hi);

          fVec res0 = (g0 / (v_one + g0.neg().exp())) * u0;
          fVec res1 = (g1 / (v_one + g1.neg().exp())) * u1;
          convert_from_float_ext<scalar_t>(res0, res1).store(o_ptr + d);
        }
        for (; d < dim; ++d) {
          float g = static_cast<float>(static_cast<at::BFloat16>(static_cast<float>(g_ptr[d])));
          float u = static_cast<float>(static_cast<at::BFloat16>(static_cast<float>(u_ptr[d])));
          g = std::min(g, limit_hi);
          u = std::min(std::max(u, limit_lo), limit_hi);
          o_ptr[d] = static_cast<scalar_t>((g / (1.0f + std::exp(-g))) * u);
        }
      }
    });
  } else {
    at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const scalar_t* __restrict__ row_in = in_base + i * in_s0;
        scalar_t* __restrict__ row_out = out_base + i * out_s0;
        for (int64_t d = 0; d < dim; ++d) {
          float g = static_cast<float>(static_cast<at::BFloat16>(static_cast<float>(row_in[d * in_s1])));
          float u = static_cast<float>(static_cast<at::BFloat16>(static_cast<float>(row_in[(d + dim) * in_s1])));
          g = std::min(g, limit_hi);
          u = std::min(std::max(u, limit_lo), limit_hi);
          row_out[d * out_s1] = static_cast<scalar_t>((g / (1.0f + std::exp(-g))) * u);
        }
      }
    });
  }
}

}  // anonymous namespace

void silu_and_mul_clamp_cpu(const at::Tensor& input, at::Tensor& output, double swiglu_limit) {
  TORCH_CHECK(input.device().is_cpu() && output.device().is_cpu(), "silu_and_mul_clamp_cpu expects CPU tensors");
  CHECK_DIM(2, input);
  CHECK_DIM(2, output);
  const int64_t num_tokens = output.size(0);
  const int64_t dim = output.size(1);
  TORCH_CHECK(
      input.size(0) == num_tokens && input.size(1) == 2 * dim,
      "Input last dimension must be twice the output last dimension");
  const auto st = input.scalar_type();
  TORCH_CHECK(
      (st == at::kHalf || st == at::kBFloat16) && output.scalar_type() == st,
      "CPU clamped SwiGLU requires matching FP16/BF16 input and output");

  if (num_tokens == 0 || dim == 0) {
    return;
  }

  // Match CUDA SiluAndMulClampKernel: clamp occurs in BF16 precision before FP32 SiLU * up.
  const float limit_hi = static_cast<float>(static_cast<at::BFloat16>(static_cast<float>(swiglu_limit)));
  const float limit_lo = static_cast<float>(static_cast<at::BFloat16>(-static_cast<float>(swiglu_limit)));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "silu_and_mul_clamp_cpu", [&] {
    silu_and_mul_clamp_kernel_impl<scalar_t>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        num_tokens,
        dim,
        input.stride(0),
        input.stride(1),
        output.stride(0),
        output.stride(1),
        limit_lo,
        limit_hi);
  });
}
