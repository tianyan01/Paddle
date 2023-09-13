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
void AssignPosCompute(const phi::GPUContext &dev_ctx,
                      framework::Tensor* cum_count, // (counter number) int32 | int64
                      framework::Tensor* numbers, // (batch_size * seq_len, topk) int32
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
}