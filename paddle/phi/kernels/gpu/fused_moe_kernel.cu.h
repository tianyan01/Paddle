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
#include <cub/cub.cuh>
#include "paddle/phi/kernels/fused_moe_kernel.h"

DECLARE_bool(avoid_op_randomness);

namespace phi {
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
void AssignPosCompute(
    const phi::GPUContext& dev_ctx,
    framework::Tensor* cum_count,  // (counter number) int32 | int64
    framework::Tensor* numbers,    // (batch_size * seq_len, topk) int32
    framework::Tensor* out,
    const int eff_num_len) {
  auto place = dev_ctx.GetPlace();
  auto numel = numbers->numel();
  T* cum_data = const_cast<T*>(cum_count->data<T>());

  framework::DDim out_dims = phi::make_ddim({eff_num_len});
  auto out_data = out->mutable_data<T>(out_dims, place);

  const T* num_data = numbers->data<T>();

  int blocks = NumBlocks(numel);
  int threads = kNumCUDAThreads;

  AssignPos<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
      cum_data, num_data, out_data, numel);
}

template <typename T>
__global__ void AssignInsAndPos(T* cum_count,
                                const T* numbers,
                                T* out,
                                int64_t limit,
                                const int topk,
                                T* ins_out) {
  CUDA_KERNEL_LOOP(i, limit) {
    auto& number_idx = numbers[i];
    if (number_idx > -1) {
      T p = platform::CudaAtomicAdd(cum_count + number_idx, -1);
      out[p - 1] = static_cast<T>(i);
      ins_out[p - 1] = static_cast<T>(i / topk);
    }
  }
}

template <typename T>
void AssignInsAndPosCompute(
    const phi::GPUContext& dev_ctx,
    phi::DenseTensor* cum_count,      // (counter number) int32 | int64
    const phi::DenseTensor* numbers,  // (batch_size * seq_len, topk) int32
    phi::DenseTensor* out,
    const int eff_num_len,
    const int topk,
    phi::DenseTensor* ins_out) {
  auto place = dev_ctx.GetPlace();
  auto numel = numbers->numel();
  T* cum_data = const_cast<T*>(cum_count->data<T>());

  framework::DDim out_dims = phi::make_ddim({eff_num_len});
  auto out_data = out->mutable_data<T>(out_dims, place);

  const T* num_data = numbers->data<T>();

  int blocks = NumBlocks(numel);
  int threads = kNumCUDAThreads;

  if (topk > 1) {
    AssignInsAndPos<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
        cum_data, num_data, out_data, numel, topk, ins_out->data<T>());
  } else {
    AssignPos<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
        cum_data, num_data, out_data, numel);
    ins_out = out;
  }
}
template <typename T>
void CumsumTensorValue(const phi::GPUContext& dev_ctx,
                       const phi::DenseTensor& in,
                       phi::DenseTensor* out,
                       const int out_offset = 0) {
  const T* d_in = in.data<T>();
  T* d_out = &out->data<T>()[out_offset];
  int num_items = in.numel();
  auto stream = dev_ctx.stream();

  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(
      NULL, temp_storage_bytes, d_in, d_out, num_items, stream);
  // Allocate temporary storage for inclusive prefix sum
  void* d_temp_storage = dev_ctx.GetWorkSpacePtr(temp_storage_bytes);
  // Run inclusive prefix sum
  cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}
}  // namespace phi