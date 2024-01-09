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

#include "paddle/fluid/operators/fused/cusparseLt.h"
#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/platform/dynload/cusparseLt.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11040)
#include "paddle/phi/kernels/funcs/blas/blaslt_impl.cu.h"
#endif
#if defined(PADDLE_WITH_CUTLASS)
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"
#endif
DECLARE_bool(is_w_prune);
namespace dyl = paddle::platform::dynload;
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
// support gemm-nt and gemm-nn, which is used in fused_attention_op.
template <typename T>
class AttnMatMul {
 public:
  // (m, n, k) = bsz_seq, output_size, input_size
  AttnMatMul(const phi::GPUContext& dev_ctx,
             bool transA,
             bool transB,
             int bsz_seq,
             int output_size,
             int input_size,
             bool compute_bias)
      : dev_ctx_(dev_ctx),
        transA_(transA),
        transB_(transB),
        bsz_seq_(bsz_seq),
        output_size_(output_size),
        input_size_(input_size),
        compute_bias_(compute_bias) {}

  ~AttnMatMul() {}

  void ComputeForward(const framework::Tensor* weight,
                      const framework::Tensor* input,
                      const framework::Tensor* bias,
                      framework::Tensor* output,
                      framework::Tensor* bias_out) {
    // Note: for blas.GEMM API in Paddle, it treats all inputs as row-major.
    // here: (transa, transb): nt, input * weight.
#if (CUDA_VERSION >= 11040)
    if (compute_bias_ && bias != nullptr) {
      phi::funcs::LinearWithCublasLt<T>::Run(
          dev_ctx_,
          input,                                      // x
          weight,                                     // y
          bias_out,                                   // out
          static_cast<const void*>(bias->data<T>()),  // bias
          nullptr,
          bsz_seq_,      // M   bsz_seq
          output_size_,  // N   output_size
          input_size_,   // K   input_size
          transA_,
          transB_,
          phi::funcs::MatmulFusedType::kMatmulBias);
      return;
    }
#endif
    CBLAS_TRANSPOSE transA = transA_ ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transB = transB_ ? CblasTrans : CblasNoTrans;
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);

    // (m, n, k) = bsz_seq, output_size, input_size, (input, weight, out)
    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx_);
    blas.GEMM(transA,
              transB,
              bsz_seq_,
              output_size_,
              input_size_,
              alpha,
              input->data<T>(),
              weight->data<T>(),
              beta,
              output->data<T>());
    if (compute_bias_ && bias != nullptr) {
      // bias_out = output + bias
      std::vector<const Tensor*> ins = {output, bias};
      std::vector<Tensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
    }
  }

  void ComputeBackward(const framework::Tensor* input,
                       const framework::Tensor* weight,
                       const framework::Tensor* d_output,
                       framework::Tensor* d_input,
                       framework::Tensor* d_weight,
                       framework::Tensor* d_bias,
                       bool use_addto = false) {
    T alpha = static_cast<T>(1.0);
    T beta_dA = use_addto ? static_cast<T>(1.0) : static_cast<T>(0.0);
    T beta_dB = static_cast<T>(0.0);

    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx_);
    if (!transA_) {
      // forward: gemm-nt
      if (transB_) {
        // backward: gemm-tn, dB = (dC)^T * A
        if (d_weight) {
          int dB_m = output_size_;
          int dB_n = input_size_;
          int dB_k = bsz_seq_;

          T* dB_output_ptr = d_weight->data<T>();
          blas.GEMM(CblasTrans,
                    CblasNoTrans,
                    dB_m,
                    dB_n,
                    dB_k,
                    alpha,
                    d_output->data<T>(),
                    input->data<T>(),
                    beta_dB,
                    dB_output_ptr);
        }

        // backward: gemm-nn, dA = dC * B
        if (d_input) {
          int dA_m = bsz_seq_;
          int dA_n = input_size_;
          int dA_k = output_size_;

          T* dA_output_ptr = d_input->data<T>();
          blas.GEMM(CblasNoTrans,
                    CblasNoTrans,
                    dA_m,
                    dA_n,
                    dA_k,
                    alpha,
                    d_output->data<T>(),
                    weight->data<T>(),
                    beta_dA,
                    dA_output_ptr);
        }
      } else {  // fw: gemm-nn
        // backward: gemm-tn, dB = A^T * dC
        if (d_weight) {
          int dB_m = input_size_;
          int dB_n = output_size_;
          int dB_k = bsz_seq_;

          T* dB_output_ptr = d_weight->data<T>();
          blas.GEMM(CblasTrans,
                    CblasNoTrans,
                    dB_m,
                    dB_n,
                    dB_k,
                    alpha,
                    input->data<T>(),
                    d_output->data<T>(),
                    beta_dB,
                    dB_output_ptr);
        }

        // backward: gemm-nt, dA = dC * B^T
        if (d_input) {
          int dA_m = bsz_seq_;
          int dA_n = input_size_;
          int dA_k = output_size_;

          T* dA_output_ptr = d_input->data<T>();
          blas.GEMM(CblasNoTrans,
                    CblasTrans,
                    dA_m,
                    dA_n,
                    dA_k,
                    alpha,
                    d_output->data<T>(),
                    weight->data<T>(),
                    beta_dA,
                    dA_output_ptr);
        }
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "AttnMatMul wrapper do not support (transA=T, transB=T/N)"
          "parameters."));
    }
    if (compute_bias_ && d_bias) {
      // reduce: {0, 1, 2, 3, 4} -> {2, 3, 4} or {0, 1, 2} -> {2} or {0,1,2,3}
      // -> {3} or {0,1,2,3,4} -> {3,4}
      const auto input_dims = d_output->dims();
      const auto output_dims = d_bias->dims();
      bool support_case_1 =
          (input_dims.size() == 5 && output_dims.size() == 3 &&
           (input_dims[2] == output_dims[0]) &&
           (input_dims[3] == output_dims[1]) &&
           (input_dims[4] == output_dims[2]));
      bool support_case_2 =
          (input_dims.size() == 3 && output_dims.size() == 1 &&
           (input_dims[2] == output_dims[0]));
      bool support_case_3 =
          (input_dims.size() == 4 && output_dims.size() == 1 &&
           input_dims[3] == output_dims[0]);
      bool support_case_4 =
          (input_dims.size() == 5 && output_dims.size() == 2 &&
           input_dims[3] == output_dims[0] && input_dims[4] == output_dims[1]);

      gpuStream_t stream = dev_ctx_.stream();
      if (support_case_1 || support_case_2) {
        TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
            dev_ctx_,
            *d_output,
            d_bias,
            kps::IdentityFunctor<T>(),
            {0, 1},
            stream);
      } else if (support_case_3 || support_case_4) {
        TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
            dev_ctx_,
            *d_output,
            d_bias,
            kps::IdentityFunctor<T>(),
            {0, 1, 2},
            stream);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Only support reduce when the input dims are [0,1,2,3,4] and "
            "output is [2,3,4]"
            "or input is [0,1,2] and output is [2]."));
      }
    }
  }

 private:
  const phi::GPUContext& dev_ctx_;

  bool transA_;
  bool transB_;

  int bsz_seq_;
  int output_size_;
  int input_size_;

  int compute_bias_;
};

template <typename T>
class AttnMatMulWeightOnly {
#if defined(PADDLE_WITH_CUTLASS)
  using InputType = typename phi::PDDataTypeTraits<T>::DataType;
  using GemRunnerInt8 = phi::CutlassFpAIntBGemmRunner<InputType, uint8_t>;
  using GemRunnerInt4 =
      phi::CutlassFpAIntBGemmRunner<InputType, cutlass::uint4b_t>;
#endif
 public:
  // (m, n, k) = bsz_seq, output_size, input_size
  AttnMatMulWeightOnly(const phi::GPUContext& dev_ctx, bool is_uint4)
      : dev_ctx_(dev_ctx), is_uint4_(is_uint4) {}

  ~AttnMatMulWeightOnly() {}
  // get activation
  int GetActivation(const std::string& act_method) {
#if defined(PADDLE_WITH_CUTLASS)
    return static_cast<int>(phi::getActivationType(act_method));
#else
    return 0;
#endif
  }
  void Linear(const phi::DenseTensor& x,
              const phi::DenseTensor& weight,
              const phi::DenseTensor* bias,
              const phi::DenseTensor& weight_scale,
              const int m,
              const int n,
              const int k,
              const int& act_method,  // none, gelu, relu
              phi::DenseTensor* out) {
#if defined(PADDLE_WITH_CUTLASS)
    const T* x_data = x.data<T>();
    const int8_t* weight_data = weight.data<int8_t>();
    const T* bias_data = bias ? bias->data<T>() : nullptr;
    const T* weight_scale_data = weight_scale.data<T>();
    T* out_data = out->data<T>();

    if (is_uint4_) {
      int mixgemm_max_size = std::max(m, k);

      int64_t mixgemm_workspace_size_bytes =
          mixed_gemm_runner_int4_.getWorkspaceSize(
              m, mixgemm_max_size, mixgemm_max_size);

      char* mixgemm_workspace_data = reinterpret_cast<char*>(
          dev_ctx_.template GetWorkSpacePtr(mixgemm_workspace_size_bytes));
      if (bias_data) {
        mixed_gemm_runner_int4_.gemm_bias_act(
            reinterpret_cast<const InputType*>(x_data),
            reinterpret_cast<const cutlass::uint4b_t*>(weight_data),
            reinterpret_cast<const InputType*>(weight_scale_data),
            reinterpret_cast<const InputType*>(bias_data),
            reinterpret_cast<InputType*>(out_data),
            m,
            n,
            k,
            static_cast<phi::ActivationType>(act_method),
            mixgemm_workspace_data,
            mixgemm_workspace_size_bytes,
            dev_ctx_.stream());
      } else {
        mixed_gemm_runner_int4_.gemm(
            reinterpret_cast<const InputType*>(x_data),
            reinterpret_cast<const cutlass::uint4b_t*>(weight_data),
            reinterpret_cast<const InputType*>(weight_scale_data),
            reinterpret_cast<InputType*>(out_data),
            m,
            n,
            k,
            mixgemm_workspace_data,
            mixgemm_workspace_size_bytes,
            dev_ctx_.stream());
      }
    } else {
      int mixgemm_max_size = std::max(m, k);
      int64_t mixgemm_workspace_size_bytes =
          mixed_gemm_runner_int8_.getWorkspaceSize(
              m, mixgemm_max_size, mixgemm_max_size);
      char* mixgemm_workspace_data = reinterpret_cast<char*>(
          dev_ctx_.template GetWorkSpacePtr(mixgemm_workspace_size_bytes));
      if (bias_data) {
        mixed_gemm_runner_int8_.gemm_bias_act(
            reinterpret_cast<const InputType*>(x_data),
            reinterpret_cast<const uint8_t*>(weight_data),
            reinterpret_cast<const InputType*>(weight_scale_data),
            reinterpret_cast<const InputType*>(bias_data),
            reinterpret_cast<InputType*>(out_data),
            m,
            n,
            k,
            static_cast<phi::ActivationType>(act_method),
            mixgemm_workspace_data,
            mixgemm_workspace_size_bytes,
            dev_ctx_.stream());
      } else {
        mixed_gemm_runner_int8_.gemm(
            reinterpret_cast<const InputType*>(x_data),
            reinterpret_cast<const uint8_t*>(weight_data),
            reinterpret_cast<const InputType*>(weight_scale_data),
            reinterpret_cast<InputType*>(out_data),
            m,
            n,
            k,
            mixgemm_workspace_data,
            mixgemm_workspace_size_bytes,
            dev_ctx_.stream());
      }
    }
#else
    PADDLE_THROW(platform::errors::InvalidArgument(
        "this machine not support weight only"));
#endif
  }

 private:
  const phi::GPUContext& dev_ctx_;
#if defined(PADDLE_WITH_CUTLASS)
  GemRunnerInt8 mixed_gemm_runner_int8_;
  GemRunnerInt4 mixed_gemm_runner_int4_;
#endif
  bool is_uint4_ = false;
};

template <typename T>
class AttnCuSparseMatMul {
 public:
  // (m, n, k) = bsz_seq, output_size, input_size
  // A(m*k) * B(k*n) = C(m*n)
  AttnCuSparseMatMul(const phi::GPUContext& dev_ctx,
                     int bsz_seq,
                     int output_size,
                     int input_size,
                     bool compute_bias)
      : dev_ctx_(dev_ctx),
        m(bsz_seq),      // m
        n(output_size),  // n
        k(input_size),   // k
        compute_bias_(compute_bias) {
    if (FLAGS_is_w_prune) {
      handle = dev_ctx.cusparselt_handle();
      streams[0] = dev_ctx.stream();
      // cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);
      // cudaStreamCreateWithFlags(&streams[2], cudaStreamNonBlocking);

      workspace_size = 0;
      if (std::is_same<T, phi::dtype::float16>::value) {
        a_type = CUDA_R_16F;
        b_type = CUDA_R_16F;
        c_type = CUDA_R_16F;
        compute_type = CUSPARSE_COMPUTE_16F;
      } else if (std::is_same<T, float>::value) {
        a_type = CUDA_R_32F;
        b_type = CUDA_R_32F;
        c_type = CUDA_R_32F;
        compute_type = CUSPARSE_COMPUTE_TF32;
      } else {
        PADDLE_ENFORCE_EQ(
            1,
            0,
            platform::errors::Fatal(
                "AttnCuSparseMatMul only suport CUDA_R_16F and CUDA_R_32F"));
      }

      CHECK_CUSPARSE(dyl::cusparseLtDenseDescriptorInit(
          handle, &matA, m, k, k, alignment, a_type, CUSPARSE_ORDER_ROW));
      CHECK_CUSPARSE(dyl::cusparseLtStructuredDescriptorInit(
          handle,
          &matB,
          k,
          n,
          n,
          alignment,
          b_type,
          CUSPARSE_ORDER_ROW,
          CUSPARSELT_SPARSITY_50_PERCENT));
      CHECK_CUSPARSE(dyl::cusparseLtDenseDescriptorInit(
          handle, &matC, m, n, n, alignment, c_type, CUSPARSE_ORDER_ROW));
      // matmul, algorithm selection, and plan initialization
      CHECK_CUSPARSE(
          dyl::cusparseLtMatmulDescriptorInit(handle,
                                              &matmul,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &matA,
                                              &matB,
                                              &matC,
                                              &matC,
                                              compute_type));
      CHECK_CUSPARSE(dyl::cusparseLtMatmulAlgSelectionInit(
          handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));

      CHECK_CUSPARSE(
          dyl::cusparseLtMatmulPlanInit(handle, &plan, &matmul, &alg_sel));
    }
  }

  void reset(int bsz_seq) {
    if (m != bsz_seq) {
      m = bsz_seq;
      CHECK_CUSPARSE(dyl::cusparseLtDenseDescriptorInit(
          handle, &matA, m, k, k, alignment, a_type, CUSPARSE_ORDER_ROW));
      CHECK_CUSPARSE(dyl::cusparseLtDenseDescriptorInit(
          handle, &matC, m, n, n, alignment, c_type, CUSPARSE_ORDER_ROW));
      CHECK_CUSPARSE(
          dyl::cusparseLtMatmulDescriptorInit(handle,
                                              &matmul,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &matA,
                                              &matB,
                                              &matC,
                                              &matC,
                                              compute_type));
      CHECK_CUSPARSE(dyl::cusparseLtMatmulAlgSelectionInit(
          handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
      CHECK_CUSPARSE(
          dyl::cusparseLtMatmulPlanInit(handle, &plan, &matmul, &alg_sel));
      algid_param = nullptr;
    }
  }

  ~AttnCuSparseMatMul() {
    if (FLAGS_is_w_prune) {
      CHECK_CUSPARSE(dyl::cusparseLtMatDescriptorDestroy(&matA));
      CHECK_CUSPARSE(dyl::cusparseLtMatDescriptorDestroy(&matB));
      CHECK_CUSPARSE(dyl::cusparseLtMatDescriptorDestroy(&matC));
      CHECK_CUSPARSE(dyl::cusparseLtMatmulPlanDestroy(&plan));
      // cudaStreamDestroy(streams[1]);
      // cudaStreamDestroy(streams[2]);
    }
  }

  void ComputeForward(const void* weight,
                      const framework::Tensor* input,
                      const framework::Tensor* bias,
                      framework::Tensor* output,
                      framework::Tensor* bias_out) {
    // Note: for blas.GEMM API in Paddle, it treats all inputs as row-major.
    // here: (transa, transb): nt, input * weight.

    float alpha = 1.0;
    float beta = 0.0;
    if (algid_param == nullptr) {
      algid_param = CuSparseLtAlgoCache::Instance().CuSparseLtAlgoSelect(
          handle,
          &plan,
          &alg_sel,
          m,
          n,
          k,
          b_type,
          &alpha,
          input->data<T>(),
          weight,
          &beta,
          output->data<T>(),
          output->data<T>(),
          streams,
          1);
      CHECK_CUSPARSE(
          dyl::cusparseLtMatmulAlgSetAttribute(handle,
                                               &alg_sel,
                                               CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                               &(algid_param->id),
                                               sizeof(algid_param->id)));
      if (algid_param->is_split_k > 1) {
        CHECK_CUSPARSE(dyl::cusparseLtMatmulAlgSetAttribute(
            handle,
            &alg_sel,
            CUSPARSELT_MATMUL_SPLIT_K,
            &(algid_param->is_split_k),
            sizeof(algid_param->is_split_k)));
        CHECK_CUSPARSE(dyl::cusparseLtMatmulAlgSetAttribute(
            handle,
            &alg_sel,
            CUSPARSELT_MATMUL_SPLIT_K_MODE,
            &(algid_param->split_k_mode),
            sizeof(algid_param->split_k_mode)));
        CHECK_CUSPARSE(dyl::cusparseLtMatmulAlgSetAttribute(
            handle,
            &alg_sel,
            CUSPARSELT_MATMUL_SPLIT_K_BUFFERS,
            &(algid_param->split_k_buffer),
            sizeof(algid_param->split_k_buffer)));
      }
      CHECK_CUSPARSE(
          dyl::cusparseLtMatmulPlanInit(handle, &plan, &matmul, &alg_sel));
      CHECK_CUSPARSE(
          dyl::cusparseLtMatmulGetWorkspace(handle, &plan, &workspace_size));
      workspace_size = (workspace_size / 8) + 1;  // to int8
      // VLOG(0) << " workspace_size  is " << workspace_size;
      work_place_tensor.Resize({{(int64_t)workspace_size}});
      dev_ctx_.Alloc<int8_t>(&work_place_tensor,
                             work_place_tensor.numel() * sizeof(int8_t));
    }
    // cusparseLtMatmulAlgSetAttribute and cusparseLtMatmulPlanInit
    // cusparseLtMatmulGetWorkspace before this
    CHECK_CUSPARSE(dyl::cusparseLtMatmul(
        handle,
        &plan,
        &alpha,
        input->data<T>(),
        weight,
        &beta,
        output->data<T>(),
        output->data<T>(),
        reinterpret_cast<void*>(work_place_tensor.data<int8_t>()),
        streams,
        1));
    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const Tensor*> ins = {output, bias};
      std::vector<Tensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
    }
  }

 private:
  cusparseLtMatDescriptor_t matA;
  cusparseLtMatDescriptor_t matB;
  cusparseLtMatDescriptor_t matC;
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulPlan_t plan;
  const phi::GPUContext& dev_ctx_;
  int m;
  int n;
  int k;
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
};

}  // namespace operators
}  // namespace paddle
