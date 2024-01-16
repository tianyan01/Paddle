/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
// This file has been adapted from FasterTransformer file:
// https://github.com/NVIDIA/FasterTransformer/blob/v4.0/fastertransformer/cuda/masked_multihead_attention.cu
// We add License in the head.

#pragma once
// #include <type_traits>
#include "paddle/fluid/operators/fused/fused_multi_transformer_op.h"
#include "paddle/phi/kernels/gpu/fused_moe_kernel.cu.h"
#include "paddle/fluid/operators/fused/attn_gemm_int8.h"

namespace paddle {
namespace operators {

using Tensor = Tensor;
using phi::backends::gpu::GpuLaunchConfig;
// This function is used to execute GEMM, with input and output's types are T
// and INT8.
template <typename T>
void MatMulTToINT8(const phi::GPUContext& dev_ctx,
                   const Tensor* weight,
                   const float quant_in_scale,
                   const Tensor* input,
                   Tensor* input_tmp,
                   Tensor* output,
                   int m,
                   int n,
                   int k,
                   Tensor* workspace = nullptr,
                   const int quant_round_type = 1,
                   const float quant_max_bound = 127.0,
                   const float quant_min_bound = -127.0) {
  cublasLtHandle_t lt_handle = dev_ctx.cublaslt_handle();
  auto helper = std::make_unique<CublasLtHelper<int32_t>>(m, k, n, lt_handle);
  quantize_kernel_launcher<T>(input->data<T>(),
                              input_tmp->data<int8_t>(),
                              quant_in_scale,
                              m,
                              k,
                              quant_round_type,
                              quant_max_bound,
                              quant_min_bound,
                              dev_ctx.stream());

  helper->GEMM(input_tmp->data<int8_t>(),
               weight->data<int8_t>(),
               output->data<int32_t>(),
               dev_ctx.stream(),
               (void*)workspace->data<int8_t>(),
               workspace->numel());
}

template <typename T>
void MatMulINT8ToT(const phi::GPUContext& dev_ctx,
                   const Tensor* weight,
                   const float quant_in_scale,
                   const Tensor* input,
                   const Tensor* bias,
                   Tensor* output,
                   Tensor* output_tmp,
                   Tensor* bias_out,
                   const Tensor* dequant_out_scale,
                   int m,
                   int n,
                   int k,
                   bool compute_bias,
                   Tensor* workspace = nullptr) {
  cublasLtHandle_t lt_handle = dev_ctx.cublaslt_handle();
  auto helper = std::make_unique<CublasLtHelper<int32_t>>(m, k, n, lt_handle);
  auto gpu_config = std::make_unique<GpuLaunchConfig>(
      phi::backends::gpu::GetGpuLaunchConfig1D(
          dev_ctx, m * n, DequantKernelVecSize));

  helper->GEMM(input->data<int8_t>(),
               weight->data<int8_t>(),
               output_tmp->data<int32_t>(),
               dev_ctx.stream(),
               (void*)workspace->data<int8_t>(),
               workspace->numel());

  dequantize_kernel_launcher<int32_t, T>(output_tmp->data<int32_t>(),
                                output->data<T>(),
                                m,
                                n,
                                dev_ctx.stream(),
                                gpu_config.get(),
                                quant_in_scale,
                                dequant_out_scale->data<float>());

  if (compute_bias) {
    // bias_out = output + bias
    std::vector<const Tensor*> ins = {output, bias};
    std::vector<Tensor*> outs = {bias_out};
    phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, -1, phi::funcs::AddFunctor<T>());
  }
}

} // operators
} // paddle
