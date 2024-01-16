// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/collective/global_gather_op.h"
#include "paddle/fluid/operators/collective/global_scatter_op.h"
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"
#include "paddle/fluid/operators/layer_norm_kernel.cu.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/bmm_kernel.h"
#include "paddle/phi/kernels/cum_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/functors.h"
#include "paddle/phi/kernels/index_select_kernel.h"
#include "paddle/phi/kernels/number_count_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/scatter_kernel.h"
#include "paddle/phi/kernels/top_k_kernel.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/ProcessGroupNCCL.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif
#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11040)
#include "paddle/phi/kernels/funcs/blas/blaslt_impl.cu.h"
#endif
namespace phi {
using Tensor = DenseTensor;
namespace framework = paddle::framework;
namespace platform = paddle::platform;

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
static ncclComm_t GetNCCLCommRanks(const Place& place,
                                   const int ring_id,
                                   int* nranks) {
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
  if (map->has(ring_id)) {
    auto pg_nccl =
        static_cast<paddle::distributed::ProcessGroupNCCL*>(map->get(ring_id));
    *nranks = pg_nccl->GetSize();
    return pg_nccl->GetNCCLComm(place);
  } else {
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    *nranks = comm->nranks();
    return comm->comm();
  }
}
#endif

template <typename T>
static void AllReduce(phi::DenseTensor& tensor,  // NOLINT
                      const int ring_id,
                      const int count,
                      const phi::GPUContext& ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto place = ctx.GetPlace();
  int nranks = 0;
  auto comm = GetNCCLCommRanks(place, ring_id, &nranks);

  auto dtype =
      platform::ToNCCLDataType(framework::TransToProtoVarType(tensor.dtype()));
  int64_t numel = tensor.numel();
  const void* sendbuff = tensor.data<T>();
  void* recvbuff = tensor.mutable_data<T>(place);
  auto stream = ctx.stream();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
      sendbuff, recvbuff, count, dtype, ncclSum, comm, stream));
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

class NCCLMoECollective {
 public:
  NCCLMoECollective(const phi::GPUContext& ctx,
                    const int ring_id,
                    const int num_expert)
      : dev_ctx_(ctx), moe_ring_id_(ring_id), num_expert_(num_expert) {
    place_ = ctx.GetPlace();
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    if (moe_ring_id_ >= 0) {
      auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
      if (map->has(ring_id)) {
        auto pg_nccl = static_cast<paddle::distributed::ProcessGroupNCCL*>(
            map->get(ring_id));
        nranks_ = pg_nccl->GetSize();
        comm_ = pg_nccl->GetNCCLComm(place_);
      } else {
        auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place_);
        nranks_ = comm->nranks();
        comm_ = comm->comm();
      }
    }
#endif
    total_expert_ = nranks_ * num_expert;
    expert_offsets_.resize(total_expert_);
    cpu_local_count_.resize(total_expert_);
    cpu_global_count_.resize(total_expert_);
  }

  template <typename T>
  void AllReduce(phi::DenseTensor& tensor) {  // NOLINT
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto dtype = platform::ToNCCLDataType(
        framework::TransToProtoVarType(tensor.dtype()));
    int64_t numel = tensor.numel();
    const void* sendbuff = tensor.data<T>();
    void* recvbuff = tensor.mutable_data<T>(place_);
    auto stream = dev_ctx_.stream();
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
        sendbuff, recvbuff, numel, dtype, ncclSum, comm_, stream));
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
        "parallel op."));
#endif
  }

  template <typename T>
  void AllToAll(Tensor& tensor,  // NOLINT
                Tensor& out) {   // NOLINT
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto dtype = platform::ToNCCLDataType(
        framework::TransToProtoVarType(tensor.dtype()));
    int64_t send_numel = tensor.numel();  // send_numel

    auto stream = dev_ctx_.stream();

    framework::DDim x_dims = tensor.dims();
    framework::DDim out_dims(x_dims);
    PADDLE_ENFORCE_EQ(
        x_dims[0] % nranks_,
        0,
        platform::errors::InvalidArgument(
            "The first dimension size (%d) of the input tensor must be "
            "divisible by the number of ranks (%d).",
            x_dims[0],
            nranks_));
    auto send_buf = tensor.data<T>();
    auto recv_buf = out.mutable_data<T>(out_dims, place_);
    size_t offset = 0;
    send_numel /= nranks_;
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
    for (auto i = 0; i < nranks_; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
          send_buf + offset, send_numel, dtype, i, comm_, stream));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
          recv_buf + offset, send_numel, dtype, i, comm_, stream));
      offset += send_numel;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
        "parallel op."));
#endif
  }

  template <typename T>
  void AllGather(Tensor& tensor,  // NOLINT
                 Tensor& out) {   // NOLINT
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto dtype = platform::ToNCCLDataType(
        framework::TransToProtoVarType(tensor.dtype()));
    int64_t numel = tensor.numel();
    auto stream = dev_ctx_.stream();
    auto out_dims = tensor.dims();
    out_dims[0] *= nranks_;
    out.mutable_data<T>(out_dims, place_);
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
        tensor.data<T>(), out.data<T>(), numel, dtype, comm_, stream));
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
        "parallel op."));
#endif
  }

  template <typename T>
  void Scatter(const framework::Tensor* x,
               const phi::DenseTensor& local_expert_count,
               const phi::DenseTensor& global_expert_count,
               framework::Tensor* out) {
    // compute expert data count offset
    CopyAndSetDataOffset(local_expert_count, global_expert_count);
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

    gpuStream_t stream = dev_ctx_.stream();

    auto in_feat = x->dims()[1];
    auto recv_ptr = 0;
    auto send_buf = x->data<T>();
    auto recv_buf = out->data<T>();

    for (auto i = 0; i < num_expert_; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
      for (auto j = 0; j < nranks_; ++j) {
        int idx = i + j * num_expert_;
        if (cpu_local_count_[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
              send_buf + expert_offsets_[idx] * in_feat,
              cpu_local_count_[idx] * in_feat,
              dtype,
              j,
              comm_,
              stream));
        }
        if (cpu_global_count_[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::ncclRecv(recv_buf + recv_ptr * in_feat,
                                          cpu_global_count_[idx] * in_feat,
                                          dtype,
                                          j,
                                          comm_,
                                          stream));
          recv_ptr += cpu_global_count_[idx];
        }
      }
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
    }
#else
    PADDLE_THROW(
        platform::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
    PADDLE_THROW(
        platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
  }

  template <typename T>
  void Gather(const framework::Tensor* x, framework::Tensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

    gpuStream_t stream = dev_ctx_.stream();

    auto in_feat = x->dims()[1];

    auto send_ptr = 0;
    auto send_buf = x->data<T>();
    auto recv_buf = out->data<T>();

    for (auto i = 0; i < num_expert_; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
      for (auto j = 0; j < nranks_; ++j) {
        int idx = i + j * num_expert_;
        if (cpu_global_count_[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::ncclSend(send_buf + send_ptr * in_feat,
                                          cpu_global_count_[idx] * in_feat,
                                          dtype,
                                          j,
                                          comm_,
                                          stream));
          send_ptr += cpu_global_count_[idx];
        }
        if (cpu_local_count_[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
              recv_buf + expert_offsets_[idx] * in_feat,
              cpu_local_count_[idx] * in_feat,
              dtype,
              j,
              comm_,
              stream));
        }
      }
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
    }
#else
    PADDLE_THROW(
        platform::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
    PADDLE_THROW(
        platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
  }

 protected:
  void CopyAndSetDataOffset(const phi::DenseTensor& local_expert_count,
                            const phi::DenseTensor& global_expert_count) {
#ifdef PADDLE_WITH_CUDA
    auto stream = dev_ctx_.stream();
    phi::backends::gpu::GpuMemcpyAsync(&cpu_local_count_[0],
                                       local_expert_count.data<int64_t>(),
                                       sizeof(int64_t) * total_expert_,
                                       cudaMemcpyDeviceToHost,
                                       stream);
    phi::backends::gpu::GpuMemcpyAsync(&cpu_global_count_[0],
                                       global_expert_count.data<int64_t>(),
                                       sizeof(int64_t) * total_expert_,
                                       cudaMemcpyDeviceToHost,
                                       stream);
    phi::backends::gpu::GpuStreamSync(stream);
#endif
    expert_offsets_[0] = 0;
    for (auto i = 1; i < total_expert_; ++i) {
      expert_offsets_[i] = expert_offsets_[i - 1] + cpu_local_count_[i - 1];
    }
  }

 private:
  const phi::GPUContext& dev_ctx_;
  int moe_ring_id_ = -1;
  int nranks_ = 0;
  int num_expert_ = 0;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  ncclComm_t comm_ = nullptr;
#endif
  phi::Place place_;
  int total_expert_ = 0;
  std::vector<int64_t> cpu_local_count_;
  std::vector<int64_t> cpu_global_count_;
  std::vector<int64_t> expert_offsets_;
};

template <typename T>
void MatMulAndAddGelu(const phi::GPUContext& dev_ctx,
                      const framework::Tensor* weight,
                      const framework::Tensor* input,
                      const framework::Tensor* bias,
                      bool istransA,
                      bool istransB,
                      bool compute_bias,
                      framework::Tensor* output) {
#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11040)
  phi::funcs::LinearWithCublasLt<T>::Run(
      dev_ctx,
      input,   // x
      weight,  // y
      output,  // out
      ((compute_bias) ? static_cast<const void*>(bias->data<T>())
                      : nullptr),  // bias
      nullptr,
      input->dims()[0],   // M  bsz_seq
      weight->dims()[1],  // N  output_size
      input->dims()[1],   // K  input_size
      istransA,
      istransB,
      ((compute_bias) ? phi::funcs::MatmulFusedType::kMatmulBiasGelu
                      : phi::funcs::MatmulFusedType::kMatmulGelu));
#endif
}

template <typename T>
void MatMulAndAdd(const phi::GPUContext& dev_ctx,
                  const framework::Tensor* weight,
                  const framework::Tensor* input,
                  const framework::Tensor* bias,
                  bool istransA,
                  bool istransB,
                  bool compute_bias,
                  framework::Tensor* output,
                  framework::Tensor* bias_out) {
#if (CUDA_VERSION >= 11040)
  if (compute_bias) {
    phi::funcs::LinearWithCublasLt<T>::Run(
        dev_ctx,
        input,                                      // x
        weight,                                     // y
        bias_out,                                   // out
        static_cast<const void*>(bias->data<T>()),  // bias
        nullptr,
        input->dims()[0],   // M   bsz_seq
        weight->dims()[1],  // N   output_size
        input->dims()[1],   // K   input_size
        istransA,
        istransB,
        phi::funcs::MatmulFusedType::kMatmulBias);
    return;
  }
#endif
  // Note: for blas.GEMM API in Paddle, it treats all inputs as row-major.
  // here: (transa, transb): nt, input * weight.
  CBLAS_TRANSPOSE transA = istransA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = istransB ? CblasTrans : CblasNoTrans;
  T alpha = static_cast<T>(1.0);
  T beta = static_cast<T>(0.0);

  // (m, n, k) = bsz_seq, output_size, input_size, (input, weight, out)
  auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
  blas.GEMM(transA,
            transB,
            input->dims()[0],
            weight->dims()[1],
            input->dims()[1],
            alpha,
            input->data<T>(),
            weight->data<T>(),
            beta,
            output->data<T>());
  if (compute_bias) {
    // bias_out = output + bias
    std::vector<const framework::Tensor*> ins = {output, bias};
    std::vector<framework::Tensor*> outs = {bias_out};
    phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, -1, phi::funcs::AddFunctor<T>());
  }
}

template <typename T, typename DeviceContext>
void FusedMoeKernel(const DeviceContext& context,
                    const DenseTensor& x,
                    const DenseTensor& gate_weight,
                    const DenseTensor& gate_bias,
                    const DenseTensor& ln_scale,
                    const DenseTensor& ln_bias,
                    const std::vector<const DenseTensor*>& experts_weight1,
                    const std::vector<const DenseTensor*>& experts_bias1,
                    const std::vector<const DenseTensor*>& experts_weight2,
                    const std::vector<const DenseTensor*>& experts_bias2,
                    bool pre_layer_norm,
                    float ln_epsilon,
                    int topk,
                    int mp_size,
                    int mp_rank,
                    int num_expert,
                    int world_size,
                    int moe_ring_id,
                    bool approximate,
                    DenseTensor* out);

}  // namespace phi
