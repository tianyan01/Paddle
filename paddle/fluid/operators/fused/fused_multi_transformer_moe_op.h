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
// #include "paddle/phi/kernels/funcs/eigen/common.h"
// #include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
// #include "paddle/phi/kernels/impl/slice_kernel_impl.h"
// #include "paddle/phi/kernels/gelu_kernel.h"
// #include "paddle/fluid/operators/fused/attn_bias_add.cu.h"

namespace paddle {
namespace operators {

using Tensor = Tensor;

// template <typename T>
// void BatchedMatMulAndAdd(const phi::GPUContext& dev_ctx,
//                          const Tensor* weight,
//                          const Tensor* input,
//                          const Tensor* bias,
//                          bool istransA,
//                          bool istransB,
//                          bool compute_bias,
//                          bool is_linear1,
//                          Tensor* output,
//                          Tensor* bias_out) {
//   // Note: for blas.BatchedGEMM API in Paddle, it treats all inputs as row-major.
//   // for input [bsz_seqlen, dim_embed] * expert_weight [expert_num, dim_embed, dim_feedforward]
//   CBLAS_TRANSPOSE transA = istransA ? CblasTrans : CblasNoTrans;
//   CBLAS_TRANSPOSE transB = istransB ? CblasTrans : CblasNoTrans;
//   T alpha = static_cast<T>(1.0);
//   T beta = static_cast<T>(0.0);

//   // (m, n, k) = bsz_seq, output_size, input_size, (input, weight, out)
//   auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
//   const int x_ndim = input->dims().size();
//   auto M = input->dims()[x_ndim - 2];
//   auto N = weight->dims()[2];
//   auto K = input->dims()[x_ndim - 1];
//   auto out_batch_size = weight->dims()[0];
//   int64_t strideA = is_linear1 ? 0 : M * K;
//   blas.BatchedGEMM(transA,
//                    transB,
//                    M,
//                    N,
//                    K,
//                    alpha,
//                    input->data<T>(),
//                    weight->data<T>(),
//                    beta,
//                    output->data<T>(),
//                    out_batch_size,
//                    strideA,
//                    K * N);
//   if (compute_bias) {
//     // bias_out = output + bias
//     std::vector<const Tensor*> ins = {output, bias};
//     std::vector<Tensor*> outs = {bias_out};
//     phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
//         dev_ctx, ins, &outs, -1, phi::funcs::AddFunctor<T>());
//   }
// }

// template <typename T>
// void MatMulAndAdd(const phi::GPUContext& dev_ctx,
//                   const T* weight, // input & output params is data pointer
//                   const T* input,
//                   const T* bias,
//                   int M,
//                   int N,
//                   int K,
//                   bool istransA,
//                   bool istransB,
//                   bool compute_bias,
//                   T* output,
//                   T* bias_out) {
//   // Note: for blas.GEMM API in Paddle, it treats all inputs as row-major.
//   // here: (transa, transb): nt, input * weight.
//   CBLAS_TRANSPOSE transA = istransA ? CblasTrans : CblasNoTrans;
//   CBLAS_TRANSPOSE transB = istransB ? CblasTrans : CblasNoTrans;
//   T alpha = static_cast<T>(1.0);
//   T beta = static_cast<T>(0.0);
//   // input->dims()[0], // M 
//   // weight->dims()[1], // N
//   // input->dims()[1], // K
//   // (m, n, k) = bsz_seq, output_size, input_size, (input, weight, out)
//   auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
//   blas.GEMM(transA,
//             transB,
//             M,
//             N,
//             K,
//             alpha,
//             input,
//             weight,
//             beta,
//             output);
//   if (compute_bias) {
//     // bias_out = output + bias
//     // std::vector<const Tensor*> ins = {output, bias};
//     // std::vector<Tensor*> outs = {bias_out};
//     // phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
//     //     dev_ctx, ins, &outs, -1, phi::funcs::AddFunctor<T>());
//     LaunchBiasAddFwKernel(dev_ctx,
//                           M,
//                           N,
//                           output,
//                           bias,
//                           bias_out);
//   }
// }

// template <typename T,
//           size_t D,
//           int MajorType = Eigen::RowMajor,
//           typename IndexType = Eigen::DenseIndex>
// using PhiEigenTensor = phi::EigenTensor<T, D, MajorType, IndexType>;

// using Array1 = Eigen::DSizes<Eigen::DenseIndex, 1>;
// using Array2 = Eigen::DSizes<Eigen::DenseIndex, 2>;

// template <typename T>
// void Addmm(const phi::GPUContext& dev_ctx,
//            const Tensor& input, // bias
//            const Tensor& x, // input
//            const Tensor& y, // weight
//            float alpha,
//            float beta,
//            Tensor* out) {
//   auto input_dims = input.dims();
//   auto x_dims = x.dims();
//   auto y_dims = y.dims();

//   Tensor input_2d(input);
//   if (input.dims().size() == 1) {
//     input_dims = {1, input.dims()[0]};
//     input_2d.Resize(input_dims);
//   }

//   // dev_ctx.template Alloc<T>(out);
//   auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);

//   // calc broadcast dim
//   Array2 bcast_dims;
//   bcast_dims[0] = x_dims[0] / input_dims[0];
//   bcast_dims[1] = y_dims[1] / input_dims[1];
//   VLOG(3) << "bcast_dims=[" << bcast_dims[0] << "," << bcast_dims[1] << "]";
//   // broadcast using eigen
//   const Tensor& const_ref_input = input_2d;
//   auto eigen_input = PhiEigenTensor<T, 2>::From(const_ref_input);
//   auto eigen_out = PhiEigenTensor<T, 2>::From(*out);
//   auto& place = *dev_ctx.eigen_device();
//   phi::funcs::EigenBroadcast<std::decay_t<decltype(place)>, T, 2>::Eval(
//       place, eigen_out, eigen_input, bcast_dims);

//   T t_alpha = static_cast<T>(alpha);
//   T t_beta = static_cast<T>(beta);
//   blas.GEMM(false,
//             false,
//             x_dims[0],
//             y_dims[1],
//             x_dims[1],
//             t_alpha,
//             x.data<T>(),
//             x_dims[1],
//             y.data<T>(),
//             y_dims[1],
//             t_beta,
//             out->data<T>(),
//             y_dims[1]);
// }

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

  dequantize_kernel_launcher<T>(output_tmp->data<int32_t>(),
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