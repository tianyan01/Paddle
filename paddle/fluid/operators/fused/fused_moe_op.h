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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"
#include "paddle/fluid/operators/layer_norm_kernel.cu.h"
#include "paddle/fluid/operators/fused/attn_gemm.h"
#include "paddle/fluid/operators/matmul_v2_op.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/top_k_kernel.h"
#include "paddle/phi/kernels/cum_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/functors.h"
#include "paddle/phi/kernels/index_select_kernel.h"
#include "paddle/phi/kernels/scatter_kernel.h"
#include "paddle/fluid/operators/collective/global_scatter_op.h"
#include "paddle/fluid/operators/collective/global_gather_op.h"
#include "paddle/phi/kernels/bmm_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
// #include "paddle/fluid/framework/convert_utils.h"
// #include "paddle/fluid/platform/float16.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/ProcessGroupNCCL.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {
// number count
#define CEIL(_x_, _y_) (((_x_)-1) / (_y_) + 1)
#define PERTHREAD_EXPERTS 256
#define WARP_SIZE 32

const int CUDA_NUM_THREADS = 512;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void initialize_zero_kernel(T* data, const int length) {
  CUDA_KERNEL_LOOP(idx, length) { data[idx] = static_cast<T>(0); }
}

template <typename T>
__global__ void NumberCount(const T* numbers,
                            T* number_count,
                            int64_t batch_size,
                            int upper_range) {
  int res_tmp[PERTHREAD_EXPERTS] = {0};
  int expert_min = blockIdx.x * PERTHREAD_EXPERTS;
  int expert_max = expert_min + PERTHREAD_EXPERTS;
  if (expert_max > upper_range) {
    expert_max = upper_range;
  }
  for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
    T idx = numbers[i];
    if (idx == -1) {
      continue;
    }
    if (idx < expert_min || idx >= expert_max) {
      continue;
    }
    res_tmp[idx - expert_min] += 1;
  }
  for (int i = expert_min; i < expert_max; ++i) {
    int x = res_tmp[i - expert_min];
#pragma unroll
    for (int j = 1; j < WARP_SIZE; j <<= 1) {
#ifdef __HIPCC__
      x = x + __shfl_down(x, j);
#else
      x = x + __shfl_down_sync(-1u, x, j);
#endif
    }
    if (threadIdx.x % WARP_SIZE == 0) {
      platform::CudaAtomicAdd(number_count + i, x);
    }
  }
}

template <typename T>
void NumberCountCompute(const phi::GPUContext &dev_ctx,
                        framework::Tensor* numbers,
                        int upper_range,
                        framework::Tensor* out) {
  int64_t batch_size = numbers->numel();
  auto place = dev_ctx.GetPlace();

  framework::DDim out_dims = phi::make_ddim({upper_range});
  auto out_data = out->mutable_data<T>(out_dims, place);
  const T* gate_data = numbers->data<T>();

  initialize_zero_kernel<T>
      <<<GET_BLOCKS(upper_range), CUDA_NUM_THREADS, 0, dev_ctx.stream()>>>(
          out_data, upper_range);

  NumberCount<T>
      <<<CEIL(upper_range, PERTHREAD_EXPERTS), 256, 0, dev_ctx.stream()>>>(
          gate_data, out_data, batch_size, upper_range);
}

// assign pos
static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T>
__global__ void AssignPos(T* cum_count,
                          const T* numbers,
                          T* out,
                          int64_t limit) {
  CUDA_KERNEL_LOOP(i, limit) {
    int number_idx = numbers[i];
    if (number_idx > -1) {
      int p = platform::CudaAtomicAdd(cum_count + number_idx, -1);
      out[p - 1] = i;
    }
  }
}

template <typename T>
void AssignPosCompute(const phi::GPUContext &dev_ctx,
                        framework::Tensor* cum_count, // (counter number) int32 | int64
                        framework::Tensor* numbers, // (batch_size * seq_len, topk) int32
                        framework::Tensor* out) {
  auto place = dev_ctx.GetPlace();
  auto numel = numbers->numel();
  T* cum_data = const_cast<T*>(cum_count->data<T>());
  auto cum_size = cum_count->numel();

  framework::Tensor cpu_cum_count;
  int64_t cpu_eff_num_len_data = 0;
  if (platform::is_cpu_place(cum_count->place())) {
    cpu_eff_num_len_data = cum_count->data<T>()[cum_size - 1];
  } else {
    framework::TensorCopySync(
        *cum_count, platform::CPUPlace(), &cpu_cum_count);
    cpu_eff_num_len_data = cpu_cum_count.data<T>()[cum_size - 1];
  }

  framework::DDim out_dims = phi::make_ddim({cpu_eff_num_len_data});
  auto out_data = out->mutable_data<T>(out_dims, place);

  const T* num_data = numbers->data<T>();

  int blocks = NumBlocks(numel);
  int threads = kNumCUDAThreads;

  AssignPos<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
      cum_data, num_data, out_data, numel);
}

template <typename T>
struct GlobalScatterFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& ctx,
                  const framework::Tensor* x,
                  const framework::Tensor* local_count,
                  const framework::Tensor* global_count,
                  int ring_id,
                  bool use_calc_stream,
                  framework::Tensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
    // auto x = ctx.Input<framework::LoDTensor>("X");
    // auto local_count = ctx.Input<framework::LoDTensor>("local_count");
    // auto global_count = ctx.Input<framework::LoDTensor>("global_count");
    auto local_count_type =
        framework::TransToProtoVarType(local_count->dtype());
    auto global_count_type =
        framework::TransToProtoVarType(global_count->dtype());
    if (local_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in local_count."));
    }
    if (global_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in global_count."));
    }
    // auto out = ctx.Output<framework::LoDTensor>("Out");
    const int64_t* cpu_local_count_data;
    const int64_t* cpu_global_count_data;
    framework::Tensor cpu_local_count;
    if (platform::is_cpu_place(local_count->place())) {
      cpu_local_count_data = local_count->data<int64_t>();
    } else {
      framework::TensorCopySync(
          *local_count, platform::CPUPlace(), &cpu_local_count);
      cpu_local_count_data = cpu_local_count.data<int64_t>();
    }
    auto global_count_len = 0;
    framework::Tensor cpu_global_count;
    if (platform::is_cpu_place(global_count->place())) {
      cpu_global_count_data = global_count->data<int64_t>();
      global_count_len = global_count->numel();
    } else {
      framework::TensorCopySync(
          *global_count, platform::CPUPlace(), &cpu_global_count);
      cpu_global_count_data = cpu_global_count.data<int64_t>();
      global_count_len = cpu_global_count.numel();
    }

    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

    // int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for global scatter op must be non-negative.",
            ring_id));

    auto place = ctx.GetPlace();
    // HARD CODE HERE!
    // auto place = platform::CUDAPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    gpuStream_t stream = nullptr;
    if (use_calc_stream) {
      // auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      // stream = static_cast<phi::GPUContext*>(dev_ctx)->stream();
      stream = ctx.stream();
    } else {
      stream = comm->stream();
    }
    int nranks = comm->nranks();
    auto in_feat = x->dims()[1];
    auto n_expert = local_count->dims()[0] / nranks;
    int64_t fwd_count = 0;

    for (auto i = 0; i < global_count_len; ++i) {
      fwd_count += cpu_global_count_data[i];
    }
    framework::DDim out_dims = phi::make_ddim({fwd_count, in_feat});
    int64_t* expert_ptr = new int64_t[n_expert * nranks];
    expert_ptr[0] = 0;
    auto tot_experts = n_expert * nranks;
    for (auto i = 1; i < tot_experts; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
    }

    auto recv_ptr = 0;
    auto send_buf = x->data<T>();
    auto recv_buf = out->mutable_data<T>(out_dims, place);

    for (auto i = 0; i < n_expert; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
      for (auto j = 0; j < nranks; ++j) {
        int idx = i + j * n_expert;
        if (cpu_local_count_data[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::ncclSend(send_buf + expert_ptr[idx] * in_feat,
                                          cpu_local_count_data[idx] * in_feat,
                                          dtype,
                                          j,
                                          comm->comm(),
                                          stream));
        }
        if (cpu_global_count_data[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::ncclRecv(recv_buf + recv_ptr * in_feat,
                                          cpu_global_count_data[idx] * in_feat,
                                          dtype,
                                          j,
                                          comm->comm(),
                                          stream));
          recv_ptr += cpu_global_count_data[idx];
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
};

template <typename T>
struct GlobalScatterProcessGroupFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& ctx,
                  const framework::Tensor* x,
                  const framework::Tensor* local_count,
                  const framework::Tensor* global_count,
                  int ring_id,
                  bool use_calc_stream,
                  framework::Tensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
    // auto x = ctx.Input<framework::LoDTensor>("X");
    // auto local_count = ctx.Input<framework::LoDTensor>("local_count");
    // auto global_count = ctx.Input<framework::LoDTensor>("global_count");
    auto local_count_type =
        framework::TransToProtoVarType(local_count->dtype());
    auto global_count_type =
        framework::TransToProtoVarType(global_count->dtype());
    if (local_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in local_count."));
    }
    if (global_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in global_count."));
    }
    // auto out = ctx.Output<framework::LoDTensor>("Out");
    const int64_t* cpu_local_count_data;
    const int64_t* cpu_global_count_data;
    framework::Tensor cpu_local_count;
    if (platform::is_cpu_place(local_count->place())) {
      cpu_local_count_data = local_count->data<int64_t>();
    } else {
      framework::TensorCopySync(
          *local_count, platform::CPUPlace(), &cpu_local_count);
      cpu_local_count_data = cpu_local_count.data<int64_t>();
    }
    auto global_count_len = 0;
    framework::Tensor cpu_global_count;
    if (platform::is_cpu_place(global_count->place())) {
      cpu_global_count_data = global_count->data<int64_t>();
      global_count_len = global_count->numel();
    } else {
      framework::TensorCopySync(
          *global_count, platform::CPUPlace(), &cpu_global_count);
      cpu_global_count_data = cpu_global_count.data<int64_t>();
      global_count_len = cpu_global_count.numel();
    }

    // int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for global scatter op must be non-negative.",
            ring_id));

    auto place = ctx.GetPlace();
    // HARD CODE HERE!
    // auto place = platform::CUDAPlace();

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    distributed::ProcessGroup* pg = map->get(ring_id);
    int nranks = pg->GetSize();
    auto in_feat = x->dims()[1];
    auto n_expert = local_count->dims()[0] / nranks;
    int64_t fwd_count = 0;

    for (auto i = 0; i < global_count_len; ++i) {
      fwd_count += cpu_global_count_data[i];
    }
    framework::DDim out_dims = phi::make_ddim({fwd_count, in_feat});
    int64_t* expert_ptr = new int64_t[n_expert * nranks];
    expert_ptr[0] = 0;
    auto tot_experts = n_expert * nranks;
    for (auto i = 1; i < tot_experts; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
    }

    auto recv_ptr = 0;
    out->mutable_data<T>(out_dims, place);

    for (auto i = 0; i < n_expert; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
      for (auto j = 0; j < nranks; ++j) {
        int idx = i + j * n_expert;
        if (cpu_local_count_data[idx]) {
          phi::DenseTensor tmp = *x;
          pg->Send_Partial(tmp,
                           j,
                           expert_ptr[idx] * in_feat,
                           cpu_local_count_data[idx] * in_feat);
        }
        if (cpu_global_count_data[idx]) {
          pg->Recv_Partial(*out,
                           j,
                           recv_ptr * in_feat,
                           cpu_global_count_data[idx] * in_feat);
          recv_ptr += cpu_global_count_data[idx];
        }
      }
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
    }

#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
    PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif

#else
    PADDLE_THROW(
        platform::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
    PADDLE_THROW(
        platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
  }
};

template <typename T>
struct GlobalGatherFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& ctx,
                  const framework::Tensor* x,
                  const framework::Tensor* local_count,
                  const framework::Tensor* global_count,
                  int ring_id,
                  bool use_calc_stream,
                  framework::Tensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
    // auto x = ctx.Input<framework::LoDTensor>("X");
    // auto local_count = ctx.Input<framework::LoDTensor>("local_count");
    // auto global_count = ctx.Input<framework::LoDTensor>("global_count");
    auto local_count_type =
        framework::TransToProtoVarType(local_count->dtype());
    auto global_count_type =
        framework::TransToProtoVarType(global_count->dtype());
    if (local_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in local_count."));
    }
    if (global_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in global_count."));
    }
    // auto out = ctx.Output<framework::LoDTensor>("Out");
    const int64_t* cpu_local_count_data;
    const int64_t* cpu_global_count_data;
    auto local_count_len = 0;

    framework::Tensor cpu_local_count;
    if (platform::is_cpu_place(local_count->place())) {
      cpu_local_count_data = local_count->data<int64_t>();
      local_count_len = local_count->numel();
    } else {
      framework::TensorCopySync(
          *local_count, platform::CPUPlace(), &cpu_local_count);
      cpu_local_count_data = cpu_local_count.data<int64_t>();
      local_count_len = cpu_local_count.numel();
    }

    framework::Tensor cpu_global_count;
    if (platform::is_cpu_place(global_count->place())) {
      cpu_global_count_data = global_count->data<int64_t>();
    } else {
      framework::TensorCopySync(
          *global_count, platform::CPUPlace(), &cpu_global_count);
      cpu_global_count_data = cpu_global_count.data<int64_t>();
    }

    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

    // int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for global gather op must be non-negative.",
            ring_id));
    auto place = ctx.GetPlace();
    // auto place = platform::CUDAPlace();

    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    gpuStream_t stream = nullptr;
    if (use_calc_stream) {
      // auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      // stream = static_cast<phi::GPUContext*>(dev_ctx)->stream();
      stream = ctx.stream();
    } else {
      stream = comm->stream();
    }
    int nranks = comm->nranks();
    auto in_feat = x->dims()[1];
    auto n_expert = local_count->dims()[0] / nranks;

    auto fwd_count = 0;

    for (auto i = 0; i < local_count_len; ++i) {
      fwd_count += cpu_local_count_data[i];
    }
    framework::DDim out_dims = phi::make_ddim({fwd_count, in_feat});
    int64_t* expert_ptr = new int64_t[n_expert * nranks];
    expert_ptr[0] = 0;
    auto tot_experts = n_expert * nranks;
    for (auto i = 1; i < tot_experts; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
    }
    auto send_ptr = 0;
    auto send_buf = x->data<T>();
    auto recv_buf = out->mutable_data<T>(out_dims, place);

    for (auto i = 0; i < n_expert; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
      for (auto j = 0; j < nranks; ++j) {
        int idx = i + j * n_expert;
        if (cpu_global_count_data[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::ncclSend(send_buf + send_ptr * in_feat,
                                          cpu_global_count_data[idx] * in_feat,
                                          dtype,
                                          j,
                                          comm->comm(),
                                          stream));
          send_ptr += cpu_global_count_data[idx];
        }
        if (cpu_local_count_data[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::ncclRecv(recv_buf + expert_ptr[idx] * in_feat,
                                          cpu_local_count_data[idx] * in_feat,
                                          dtype,
                                          j,
                                          comm->comm(),
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
};

template <typename T>
struct GlobalGatherProcessGroupFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& ctx,
                  const framework::Tensor* x,
                  const framework::Tensor* local_count,
                  const framework::Tensor* global_count,
                  int ring_id,
                  bool use_calc_stream,
                  framework::Tensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
    // auto x = ctx.Input<framework::LoDTensor>("X");
    // auto local_count = ctx.Input<framework::LoDTensor>("local_count");
    // auto global_count = ctx.Input<framework::LoDTensor>("global_count");
    auto local_count_type =
        framework::TransToProtoVarType(local_count->dtype());
    auto global_count_type =
        framework::TransToProtoVarType(global_count->dtype());
    if (local_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in local_count."));
    }
    if (global_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in global_count."));
    }
    // auto out = ctx.Output<framework::LoDTensor>("Out");
    const int64_t* cpu_local_count_data;
    const int64_t* cpu_global_count_data;
    auto local_count_len = 0;

    framework::Tensor cpu_local_count;
    if (platform::is_cpu_place(local_count->place())) {
      cpu_local_count_data = local_count->data<int64_t>();
      local_count_len = local_count->numel();
    } else {
      framework::TensorCopySync(
          *local_count, platform::CPUPlace(), &cpu_local_count);
      cpu_local_count_data = cpu_local_count.data<int64_t>();
      local_count_len = cpu_local_count.numel();
    }

    framework::Tensor cpu_global_count;
    if (platform::is_cpu_place(global_count->place())) {
      cpu_global_count_data = global_count->data<int64_t>();
    } else {
      framework::TensorCopySync(
          *global_count, platform::CPUPlace(), &cpu_global_count);
      cpu_global_count_data = cpu_global_count.data<int64_t>();
    }

    // int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for global gather op must be non-negative.",
            ring_id));
    auto place = ctx.GetPlace();
    // auto place = platform::CUDAPlace();

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    distributed::ProcessGroup* pg = map->get(ring_id);

    int nranks = pg->GetSize();
    auto in_feat = x->dims()[1];
    auto n_expert = local_count->dims()[0] / nranks;

    auto fwd_count = 0;

    for (auto i = 0; i < local_count_len; ++i) {
      fwd_count += cpu_local_count_data[i];
    }
    framework::DDim out_dims = phi::make_ddim({fwd_count, in_feat});
    int64_t* expert_ptr = new int64_t[n_expert * nranks];
    expert_ptr[0] = 0;
    auto tot_experts = n_expert * nranks;
    for (auto i = 1; i < tot_experts; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
    }
    auto send_ptr = 0;
    out->mutable_data<T>(out_dims, place);

    for (auto i = 0; i < n_expert; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
      for (auto j = 0; j < nranks; ++j) {
        int idx = i + j * n_expert;
        if (cpu_global_count_data[idx]) {
          phi::DenseTensor tmp = *x;
          pg->Send_Partial(
              tmp, j, send_ptr * in_feat, cpu_global_count_data[idx] * in_feat);
          send_ptr += cpu_global_count_data[idx];
        }
        if (cpu_local_count_data[idx]) {
          pg->Recv_Partial(*out,
                           j,
                           expert_ptr[idx] * in_feat,
                           cpu_local_count_data[idx] * in_feat);
        }
      }
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
    }

#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
    PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif

#else
    PADDLE_THROW(
        platform::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
    PADDLE_THROW(
        platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
  }
};

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
    std::vector<const Tensor*> ins = {output, bias};
    std::vector<Tensor*> outs = {bias_out};
    phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, -1, phi::funcs::AddFunctor<T>());
  }
}

}   // namesapce operators
}   //namespace paddle