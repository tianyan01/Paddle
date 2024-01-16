/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <iostream>
#include <vector>
#include "paddle/fluid/operators/fused/cublaslt.h"
#include "paddle/fluid/operators/fused/cusparseLt.h"
#include "paddle/fluid/operators/fused/quant_dequant_kernel.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
DECLARE_bool(is_w_prune);
namespace paddle {
namespace operators {

using phi::backends::gpu::GpuLaunchConfig;

template <typename T>
class AttnMatmulINT8 {
 public:
  AttnMatmulINT8(
      const phi::GPUContext& dev_ctx, int m, int n, int k, bool compute_bias)
      : dev_ctx_(dev_ctx), m_(m), n_(n), k_(k), compute_bias_(compute_bias) {
    cublasLtHandle_t lt_handle = dev_ctx.cublaslt_handle();
    helper_ = std::make_unique<CublasLtHelper<int32_t>>(m, k, n, lt_handle);
    gpu_config_ = std::make_unique<GpuLaunchConfig>(
        phi::backends::gpu::GetGpuLaunchConfig1D(
            dev_ctx, m * n, DequantKernelVecSize));
  }
  ~AttnMatmulINT8() {}

  // This function is used to execute GEMM, with input and output's types are
  // both T.
  void ComputeForward(const phi::DenseTensor* weight,
                      const phi::DenseTensor* input,
                      phi::DenseTensor* input_tmp,
                      const phi::DenseTensor* bias,
                      phi::DenseTensor* output,
                      phi::DenseTensor* output_tmp,
                      phi::DenseTensor* bias_out,
                      const float quant_in_scale,
                      const phi::DenseTensor* dequant_out_scale,
                      phi::DenseTensor* workspace = nullptr,
                      const int quant_round_type = 1,
                      const float quant_max_bound = 127.0,
                      const float quant_min_bound = -127.0) {
    quantize_kernel_launcher<T>(input->data<T>(),
                                input_tmp->data<int8_t>(),
                                quant_in_scale,
                                m_,
                                k_,
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound,
                                dev_ctx_.stream());

    helper_->GEMM(input_tmp->data<int8_t>(),
                  weight->data<int8_t>(),
                  output_tmp->data<int32_t>(),
                  dev_ctx_.stream(),
                  (void*)workspace->data<int8_t>(),
                  workspace->numel());

    dequantize_kernel_launcher<int32_t, T>(output_tmp->data<int32_t>(),
                                           output->data<T>(),
                                           m_,
                                           n_,
                                           dev_ctx_.stream(),
                                           gpu_config_.get(),
                                           quant_in_scale,
                                           dequant_out_scale->data<float>());

    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const phi::DenseTensor*> ins = {output, bias};
      std::vector<phi::DenseTensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
      PADDLE_ENFORCE_EQ(cudaGetLastError(),
                        cudaSuccess,
                        platform::errors::Fatal(
                            "cuda error occured after computing bias. "
                            "But it does not mean this error is caused by "
                            "bias computing"));
    }
  }

  // This function is used to execute GEMM, with input and output's types are
  // both INT8.
  void ComputeForwardINT8ToINT8(const phi::DenseTensor* weight,
                                phi::DenseTensor* input,
                                const phi::DenseTensor* bias,
                                phi::DenseTensor* output,
                                phi::DenseTensor* bias_out,
                                phi::DenseTensor* workspace = nullptr) {
    helper_->GEMM(input->data<int8_t>(),
                  weight->data<int8_t>(),
                  output->data<int32_t>(),
                  dev_ctx_.stream(),
                  (void*)workspace->data<int8_t>(),
                  workspace->numel());
  }

  // This function is used to execute GEMM, with input and output's types are
  // INT8 and T.
  void ComputeForwardINT8ToT(const phi::DenseTensor* weight,
                             const float quant_in_scale,
                             phi::DenseTensor* input,
                             const phi::DenseTensor* bias,
                             phi::DenseTensor* output,
                             phi::DenseTensor* output_tmp,
                             phi::DenseTensor* bias_out,
                             const phi::DenseTensor* dequant_out_scale,
                             phi::DenseTensor* workspace = nullptr) {
    helper_->GEMM(input->data<int8_t>(),
                  weight->data<int8_t>(),
                  output_tmp->data<int32_t>(),
                  dev_ctx_.stream(),
                  (void*)workspace->data<int8_t>(),
                  workspace->numel());

    dequantize_kernel_launcher<int32_t, T>(output_tmp->data<int32_t>(),
                                           output->data<T>(),
                                           m_,
                                           n_,
                                           dev_ctx_.stream(),
                                           gpu_config_.get(),
                                           quant_in_scale,
                                           dequant_out_scale->data<float>());

    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const phi::DenseTensor*> ins = {output, bias};
      std::vector<phi::DenseTensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
      //   PADDLE_ENFORCE_EQ(cudaGetLastError(),
      //                     cudaSuccess,
      //                     platform::errors::Fatal(
      //                         "cuda error occured after computing bias. "
      //                         "But it does not mean this error is caused by "
      //                         "bias computing"));
    }
  }

  // This function is used to execute GEMM, with input and output's types are T
  // and INT8.
  void ComputeForwardTToINT8(const phi::DenseTensor* weight,
                             const float quant_in_scale,
                             const phi::DenseTensor* input,
                             phi::DenseTensor* input_tmp,
                             const phi::DenseTensor* bias,
                             phi::DenseTensor* output,
                             phi::DenseTensor* bias_out,
                             phi::DenseTensor* workspace = nullptr,
                             const int quant_round_type = 1,
                             const float quant_max_bound = 127.0,
                             const float quant_min_bound = -127.0) {
    quantize_kernel_launcher<T>(input->data<T>(),
                                input_tmp->data<int8_t>(),
                                quant_in_scale,
                                m_,
                                k_,
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound,
                                dev_ctx_.stream());

    helper_->GEMM(input_tmp->data<int8_t>(),
                  weight->data<int8_t>(),
                  output->data<int32_t>(),
                  dev_ctx_.stream(),
                  (void*)workspace->data<int8_t>(),
                  workspace->numel());
  }

 private:
  const phi::GPUContext& dev_ctx_;

  int m_;  // m
  int n_;  // n
  int k_;  // k

  int compute_bias_;
  std::unique_ptr<CublasLtHelper<int32_t>> helper_;
  std::unique_ptr<GpuLaunchConfig> gpu_config_;
};

template <typename T>
class AttnCuSparseMatmulINT8 {
 public:
  AttnCuSparseMatmulINT8(const phi::GPUContext& dev_ctx,
                         int padding_m,
                         int m,
                         int n,
                         int k,
                         bool compute_bias,
                         float alpha = 1.0f)
      : dev_ctx_(dev_ctx),
        padding_m_(padding_m),
        m_(m),
        n_(n),
        k_(k),
        compute_bias_(compute_bias),
        alpha_(alpha) {
    if (FLAGS_is_w_prune) {
      handle = dev_ctx.cusparselt_handle();
      streams[0] = dev_ctx.stream();
      // cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);
      // cudaStreamCreateWithFlags(&streams[2], cudaStreamNonBlocking);
      a_type = CUDA_R_8I;
      b_type = CUDA_R_8I;
      c_type = CUDA_R_16F;
      compute_type = CUSPARSE_COMPUTE_32I;
      workspace_size = 0;
      context = new cusparseLtContext;

      CHECK_CUSPARSE(dyl::cusparseLtDenseDescriptorInit(handle,
                                                        &(context->matA),
                                                        padding_m_,
                                                        k_,
                                                        k_,
                                                        alignment,
                                                        a_type,
                                                        CUSPARSE_ORDER_ROW));
      CHECK_CUSPARSE(dyl::cusparseLtStructuredDescriptorInit(
          handle,
          &(context->matB),
          n_,
          k_,
          k_,
          alignment,
          b_type,
          CUSPARSE_ORDER_ROW,
          CUSPARSELT_SPARSITY_50_PERCENT));
      CHECK_CUSPARSE(dyl::cusparseLtDenseDescriptorInit(handle,
                                                        &(context->matC),
                                                        padding_m_,
                                                        n_,
                                                        n_,
                                                        alignment,
                                                        c_type,
                                                        CUSPARSE_ORDER_ROW));
      // matmul, algorithm selection, and plan initialization
      CHECK_CUSPARSE(
          dyl::cusparseLtMatmulDescriptorInit(handle,
                                              &(context->matmul),
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_TRANSPOSE,
                                              &(context->matA),
                                              &(context->matB),
                                              &(context->matC),
                                              &(context->matC),
                                              compute_type));
      CHECK_CUSPARSE(
          dyl::cusparseLtMatmulAlgSelectionInit(handle,
                                                &(context->alg_sel),
                                                &(context->matmul),
                                                CUSPARSELT_MATMUL_ALG_DEFAULT));

      CHECK_CUSPARSE(dyl::cusparseLtMatmulPlanInit(
          handle, &(context->plan), &(context->matmul), &(context->alg_sel)));
      gpu_config_ = std::make_unique<GpuLaunchConfig>(
          phi::backends::gpu::GetGpuLaunchConfig1D(
              dev_ctx, m * n, DequantKernelVecSize));
    }
  }

  ~AttnCuSparseMatmulINT8() {
    if (FLAGS_is_w_prune) {
      CHECK_CUSPARSE(dyl::cusparseLtMatDescriptorDestroy(&(context->matA)));
      CHECK_CUSPARSE(dyl::cusparseLtMatDescriptorDestroy(&(context->matB)));
      CHECK_CUSPARSE(dyl::cusparseLtMatDescriptorDestroy(&(context->matC)));
      CHECK_CUSPARSE(dyl::cusparseLtMatmulPlanDestroy(&(context->plan)));
      delete context;
    }
  }

  void reset(int padding_m, int m) {
    m_ = m;
    if (padding_m_ != padding_m) {
      padding_m_ = padding_m;
      CHECK_CUSPARSE(dyl::cusparseLtDenseDescriptorInit(handle,
                                                        &(context->matA),
                                                        padding_m_,
                                                        k_,
                                                        k_,
                                                        alignment,
                                                        a_type,
                                                        CUSPARSE_ORDER_ROW));
      CHECK_CUSPARSE(dyl::cusparseLtDenseDescriptorInit(handle,
                                                        &(context->matC),
                                                        padding_m_,
                                                        n_,
                                                        n_,
                                                        alignment,
                                                        c_type,
                                                        CUSPARSE_ORDER_ROW));
      CHECK_CUSPARSE(
          dyl::cusparseLtMatmulDescriptorInit(handle,
                                              &(context->matmul),
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_TRANSPOSE,
                                              &(context->matA),
                                              &(context->matB),
                                              &(context->matC),
                                              &(context->matC),
                                              compute_type));
      CHECK_CUSPARSE(
          dyl::cusparseLtMatmulAlgSelectionInit(handle,
                                                &(context->alg_sel),
                                                &(context->matmul),
                                                CUSPARSELT_MATMUL_ALG_DEFAULT));
      CHECK_CUSPARSE(dyl::cusparseLtMatmulPlanInit(
          handle, &(context->plan), &(context->matmul), &(context->alg_sel)));
      algid_param = nullptr;
    }
  }

  // This function is used to execute GEMM, with input and output's types are
  // both T.
  inline void gemm(const framework::Tensor* input,
                   const void* weight,
                   framework::Tensor* output) {
    // Note: for blas.GEMM API in Paddle, it treats all inputs as row-major.
    // here: (transa, transb): nt, input * weight.

    float beta = 0.0;
    if (algid_param == nullptr) {
      algid_param = CuSparseLtAlgoCache::Instance().CuSparseLtAlgoSelect(
          handle,
          &(context->plan),
          &(context->alg_sel),
          padding_m_,
          n_,
          k_,
          b_type,
          &alpha_,
          input->data<int8_t>(),
          weight,
          &beta,
          output->data<phi::dtype::float16>(),
          output->data<phi::dtype::float16>(),
          streams,
          1);
      CHECK_CUSPARSE(
          dyl::cusparseLtMatmulAlgSetAttribute(handle,
                                               &(context->alg_sel),
                                               CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                               &(algid_param->id),
                                               sizeof(algid_param->id)));
      // VLOG(0) << " for " << m << " " << n << " " << k << " algid is " <<
      // algid_param->id;
      if (algid_param->is_split_k > 1) {
        VLOG(0) << "use split k";
        CHECK_CUSPARSE(dyl::cusparseLtMatmulAlgSetAttribute(
            handle,
            &(context->alg_sel),
            CUSPARSELT_MATMUL_SPLIT_K,
            &(algid_param->is_split_k),
            sizeof(algid_param->is_split_k)));
        CHECK_CUSPARSE(dyl::cusparseLtMatmulAlgSetAttribute(
            handle,
            &(context->alg_sel),
            CUSPARSELT_MATMUL_SPLIT_K_MODE,
            &(algid_param->split_k_mode),
            sizeof(algid_param->split_k_mode)));
        CHECK_CUSPARSE(dyl::cusparseLtMatmulAlgSetAttribute(
            handle,
            &(context->alg_sel),
            CUSPARSELT_MATMUL_SPLIT_K_BUFFERS,
            &(algid_param->split_k_buffer),
            sizeof(algid_param->split_k_buffer)));
      }
      CHECK_CUSPARSE(dyl::cusparseLtMatmulPlanInit(
          handle, &(context->plan), &(context->matmul), &(context->alg_sel)));
      CHECK_CUSPARSE(dyl::cusparseLtMatmulGetWorkspace(
          handle, &(context->plan), &workspace_size));
      workspace_size = (workspace_size / 8) + 1;  // to int8
      if (workspace_size >= work_place_tensor.numel()) {
        work_place_tensor.Resize({{(int64_t)workspace_size}});
        dev_ctx_.Alloc<int8_t>(&work_place_tensor,
                               work_place_tensor.numel() * sizeof(int8_t));
      }
    }
    // cusparseLtMatmulAlgSetAttribute and cusparseLtMatmulPlanInit
    // cusparseLtMatmulGetWorkspace before this
    CHECK_CUSPARSE(dyl::cusparseLtMatmul(
        handle,
        &(context->plan),
        &alpha_,
        input->data<int8_t>(),
        weight,
        &beta,
        output->data<phi::dtype::float16>(),
        output->data<phi::dtype::float16>(),
        reinterpret_cast<void*>(work_place_tensor.data<int8_t>()),
        streams,
        1));
  }

  void ComputeForward(void* weight,
                      const phi::DenseTensor* input,
                      phi::DenseTensor* input_tmp,
                      const phi::DenseTensor* bias,
                      phi::DenseTensor* output,
                      phi::DenseTensor* output_tmp,
                      phi::DenseTensor* bias_out,
                      const float quant_in_scale,
                      const phi::DenseTensor* dequant_out_scale,
                      phi::DenseTensor* workspace = nullptr,
                      const int quant_round_type = 1,
                      const float quant_max_bound = 127.0,
                      const float quant_min_bound = -127.0) {
    quantize_kernel_launcher<T>(input->data<T>(),
                                input_tmp->data<int8_t>(),
                                quant_in_scale,
                                m_,
                                k_,
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound,
                                dev_ctx_.stream());

    gemm(input_tmp, weight, output_tmp);
    dequantize_kernel_launcher<phi::dtype::float16, T>(
        output_tmp->data<phi::dtype::float16>(),
        output->data<T>(),
        m_,
        n_,
        dev_ctx_.stream(),
        gpu_config_.get(),
        quant_in_scale,
        dequant_out_scale->data<float>());

    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const phi::DenseTensor*> ins = {output, bias};
      std::vector<phi::DenseTensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
      PADDLE_ENFORCE_EQ(cudaGetLastError(),
                        cudaSuccess,
                        platform::errors::Fatal(
                            "cuda error occured after computing bias. "
                            "But it does not mean this error is caused by "
                            "bias computing"));
    }
  }

  // This function is used to execute GEMM, with input and output's types are
  // both INT8.
  void ComputeForwardINT8ToINT8(void* weight,
                                phi::DenseTensor* input,
                                const phi::DenseTensor* bias,
                                phi::DenseTensor* output,
                                phi::DenseTensor* bias_out,
                                phi::DenseTensor* workspace = nullptr) {
    gemm(input, weight, output);
  }

  // This function is used to execute GEMM, with input and output's types are
  // INT8 and T.
  void ComputeForwardINT8ToT(void* weight,
                             const float quant_in_scale,
                             phi::DenseTensor* input,
                             const phi::DenseTensor* bias,
                             phi::DenseTensor* output,
                             phi::DenseTensor* output_tmp,
                             phi::DenseTensor* bias_out,
                             const phi::DenseTensor* dequant_out_scale,
                             phi::DenseTensor* workspace = nullptr) {
    gemm(input, weight, output_tmp);
    dequantize_kernel_launcher<phi::dtype::float16, T>(
        (phi::dtype::float16*)output_tmp->data<phi::dtype::float16>(),
        output->data<T>(),
        m_,
        n_,
        dev_ctx_.stream(),
        gpu_config_.get(),
        quant_in_scale,
        dequant_out_scale->data<float>());

    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const phi::DenseTensor*> ins = {output, bias};
      std::vector<phi::DenseTensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
      //   PADDLE_ENFORCE_EQ(cudaGetLastError(),
      //                     cudaSuccess,
      //                     platform::errors::Fatal(
      //                         "cuda error occured after computing bias. "
      //                         "But it does not mean this error is caused by "
      //                         "bias computing"));
    }
  }

  // This function is used to execute GEMM, with input and output's types are T
  // and INT8.
  void ComputeForwardTToINT8(void* weight,
                             const float quant_in_scale,
                             const phi::DenseTensor* input,
                             phi::DenseTensor* input_tmp,
                             const phi::DenseTensor* bias,
                             phi::DenseTensor* output,
                             phi::DenseTensor* bias_out,
                             phi::DenseTensor* workspace = nullptr,
                             const int quant_round_type = 1,
                             const float quant_max_bound = 127.0,
                             const float quant_min_bound = -127.0) {
    quantize_kernel_launcher<T>(input->data<T>(),
                                input_tmp->data<int8_t>(),
                                quant_in_scale,
                                m_,
                                k_,
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound,
                                dev_ctx_.stream());
    gemm(input_tmp, weight, output);
  }

  void scale(float* src, float scale_x, int n) {
    scale_launch<float>(src, scale_x, n, streams[0]);
  }

  void set_alpha(float new_alpha) { alpha_ = new_alpha; }

  struct cusparseLtContext {
    cusparseLtMatDescriptor_t matA;
    cusparseLtMatDescriptor_t matB;
    cusparseLtMatDescriptor_t matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
  };

 private:
  const phi::GPUContext& dev_ctx_;
  cusparseLtContext* context = nullptr;
  int padding_m_;
  int m_;
  int n_;
  int k_;
  gpuStream_t streams[1];
  cusparseLtHandle_t* handle = nullptr;
  CuSparseLtAlgoParam* algid_param = nullptr;
  size_t workspace_size;
  int compute_bias_;
  cudaDataType a_type;
  cudaDataType b_type;
  cudaDataType c_type;
  cusparseComputeType compute_type;
  phi::DenseTensor work_place_tensor;
  std::unique_ptr<GpuLaunchConfig> gpu_config_;
  float alpha_ = 1.0;
};

}  // namespace operators
}  // namespace paddle
