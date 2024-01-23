/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/phi/core/dense_tensor.h"
#if defined(PADDLE_WITH_CUTLASS)
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/moe_gemm_kernels_template.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/ptq_moe_gemm/ptq_moe_gemm.h"
#endif
namespace paddle {
namespace operators {

template <typename T>
class MoeExpertGemmWeightOnly {
#if defined(PADDLE_WITH_CUTLASS)
  using InputType = typename phi::PDDataTypeTraits<T>::DataType;
  using GemRunnerInt8 = phi::MoeGemmRunner<InputType, uint8_t>;
  using GemRunnerInt4 = phi::MoeGemmRunner<InputType, cutlass::uint4b_t>;
#endif
 public:
  MoeExpertGemmWeightOnly(const phi::GPUContext &dev_ctx, bool is_uint4)
      : dev_ctx_(dev_ctx), is_uint4_(is_uint4) {}

  ~MoeExpertGemmWeightOnly() {}
  void moe_gemm(const phi::DenseTensor &expert_rows_pos,
                const phi::DenseTensor &x,
                const phi::DenseTensor *expert_weights1,
                const phi::DenseTensor *expert_scales1,
                const phi::DenseTensor *expert_biases1,
                const phi::DenseTensor *expert_weights2,
                const phi::DenseTensor *expert_scales2,
                const phi::DenseTensor *expert_biases2,
                const int fwd_bsz,
                const int dim_feedforward,
                const int dim_embed,
                const int num_expert,
                const int &act_method,   // none, gelu, relu
                const int &default_act,  // none, gelu, relu
                phi::DenseTensor *expert_out1,
                phi::DenseTensor *out) {
#if defined(PADDLE_WITH_CUTLASS)
    // csum length
    const int64_t *total_rows_before_expert = expert_rows_pos.data<int64_t>();
    const T *permuted_data = x.data<T>();
    const int8_t *fc1_expert_weights = expert_weights1->data<int8_t>();
    const T *fc1_scales = expert_scales1->data<T>();
    const T *fc1_expert_biases = expert_biases1->data<T>();
    T *fc1_result = expert_out1->data<T>();

    const int8_t *fc2_expert_weights = expert_weights2->data<int8_t>();
    const T *fc2_scales = expert_scales2->data<T>();
    const T *fc2_expert_biases = expert_biases2->data<T>();
    T *fc2_result = out->data<T>();

    if (is_uint4_) {
      gemm_runner_int4_.moe_gemm_bias_act(
          reinterpret_cast<const InputType *>(permuted_data),
          reinterpret_cast<const cutlass::uint4b_t *>(fc1_expert_weights),
          reinterpret_cast<const InputType *>(fc1_scales),
          reinterpret_cast<const InputType *>(fc1_expert_biases),
          reinterpret_cast<InputType *>(fc1_result),
          const_cast<int64_t *>(total_rows_before_expert),
          fwd_bsz,
          dim_feedforward,
          dim_embed,
          num_expert,
          static_cast<phi::ActivationType>(act_method),
          dev_ctx_.stream());
      gemm_runner_int4_.moe_gemm_bias_act(
          reinterpret_cast<const InputType *>(fc1_result),
          reinterpret_cast<const cutlass::uint4b_t *>(fc2_expert_weights),
          reinterpret_cast<const InputType *>(fc2_scales),
          reinterpret_cast<const InputType *>(fc2_expert_biases),
          reinterpret_cast<InputType *>(fc2_result),
          const_cast<int64_t *>(total_rows_before_expert),
          fwd_bsz,
          dim_embed,
          dim_feedforward,
          num_expert,
          static_cast<phi::ActivationType>(default_act),
          dev_ctx_.stream());
    } else {
      gemm_runner_int8_.moe_gemm_bias_act(
          reinterpret_cast<const InputType *>(permuted_data),
          reinterpret_cast<const uint8_t *>(fc1_expert_weights),
          reinterpret_cast<const InputType *>(fc1_scales),
          reinterpret_cast<const InputType *>(fc1_expert_biases),
          reinterpret_cast<InputType *>(fc1_result),
          const_cast<int64_t *>(total_rows_before_expert),
          fwd_bsz,
          dim_feedforward,
          dim_embed,
          num_expert,
          static_cast<phi::ActivationType>(act_method),
          dev_ctx_.stream());

      gemm_runner_int8_.moe_gemm_bias_act(
          reinterpret_cast<const InputType *>(fc1_result),
          reinterpret_cast<const uint8_t *>(fc2_expert_weights),
          reinterpret_cast<const InputType *>(fc2_scales),
          reinterpret_cast<const InputType *>(fc2_expert_biases),
          reinterpret_cast<InputType *>(fc2_result),
          const_cast<int64_t *>(total_rows_before_expert),
          fwd_bsz,
          dim_embed,
          dim_feedforward,
          num_expert,
          static_cast<phi::ActivationType>(default_act),
          dev_ctx_.stream());
    }
#else
        PADDLE_THROW(platform::errors::InvalidArgument(
            "this machine not support weight only"));
#endif
  }

 private:
  const phi::GPUContext &dev_ctx_;
#if defined(PADDLE_WITH_CUTLASS)
  GemRunnerInt8 gemm_runner_int8_;
  GemRunnerInt4 gemm_runner_int4_;
#endif
  bool is_uint4_ = false;
};

// for ptq
template <typename T>
void FusedGroupedMatMul(const phi::GPUContext& dev_ctx,
                        const int8_t* weight, // int8
                        const T* input, // fp16, shape is [fwd_bsz, k]
                        Tensor* input_tmp, // int8
                        const float* quant_in_scale,
                        const T* bias, // fp16
                        T* output_deq,
                        const float* dequant_out_scale, // fp32
                        const int64_t* fwd_expert_count_cumsum, // int64
                        const int64_t* fwd_expert_count_cumsum_cpu, // cpu
                        int fwd_bsz,
                        int num_expert,
                        int m,
                        int n,
                        int k,
                        const bool do_activation,
                        const int quant_round_type = 1,
                        const float quant_max_bound = 127.0,
                        const float quant_min_bound = -127.0) {
#if defined(PADDLE_WITH_CUTLASS)
  int64_t offset = 0;
  for (int i = 0; i < num_expert; ++i) {
    int64_t cur_m = *(fwd_expert_count_cumsum_cpu + i + 1) - *(fwd_expert_count_cumsum_cpu + i);
    if (cur_m == 0) {
      continue;
    }
    quantize_kernel_launcher<T>(input + offset,
                                input_tmp->data<int8_t>() + offset,
                                *(quant_in_scale + i),
                                cur_m, // cur m
                                k,
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound,
                                dev_ctx.stream());
    offset += cur_m * k;
  };

  // group_quantize_kernel_launcher(input,
  //                               input_tmp->data<int8_t>(),
  //                               quant_in_scale,
  //                               fwd_expert_count,
  //                               num_expert,
  //                               m,
  //                               k,
  //                               quant_round_type,
  //                               quant_max_bound,
  //                               quant_min_bound,
  //                               dev_ctx.stream());

  using half_dtype = typename phi::PDDataTypeTraits<paddle::platform::float16>::DataType;
  auto moe_gemm_runner = phi::PTQMoeGemmRunner<half_dtype>();
  // int8 gemm & dequant & add biad & act(optional)
  moe_gemm_runner.moe_gemm_bias_act(input_tmp->data<int8_t>(),
                                    weight,
                                    dequant_out_scale,
                                    reinterpret_cast<const half_dtype*>(bias), // bias
                                    reinterpret_cast<half_dtype*>(output_deq),
                                    fwd_expert_count_cumsum,
                                    fwd_bsz,
                                    n,
                                    k,
                                    num_expert,
                                    do_activation,
                                    dev_ctx.stream());
#else
  PADDLE_THROW(platform::errors::InvalidArgument(
      "this machine not support FusedGroupedMatMul use cutlass"));
#endif
}

}  // namespace operators
}  // namespace paddle
