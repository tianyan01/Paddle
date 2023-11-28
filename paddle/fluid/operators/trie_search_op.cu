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

#include "paddle/fluid/operators/trie_search_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/framework/trie_manager.h"

namespace paddle {
namespace operators {

const int CUDA_NUM_THREADS = platform::PADDLE_CUDA_NUM_THREADS;
#define GET_BLOCK(N) ((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)

#define CUDA_KERNEL_LOOP(i, n)                                  \
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
#define CUDA_BLOCK(N) GET_BLOCK(N), CUDA_NUM_THREADS, 0

__global__ void MaskKernel(const int64_t len, const int64_t* lod, const int64_t* idx, int64_t C, int64_t N, float* arr) {
  CUDA_KERNEL_LOOP(i, len) {
    int low = 0;
    int high = C - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < lod[mid + 1]) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }

    int x = low * N + idx[i];
    arr[x] = 0;
  }
}

__global__ void MaskKernel1D(const int64_t total, const int64_t idx_len, const int64_t* idx, int64_t C,int64_t N, float* arr) {
  CUDA_KERNEL_LOOP(i, total) {
    int64_t c = i / idx_len;
    int64_t x = c * N + idx[i % idx_len];
    arr[x] = 0;
  }
}

template <typename T>
class TrieSearchStartCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    VLOG(3) << "TrieSearchStartCUDAKernel";
    auto s = framework::TrieManager::GetInstance();
    const framework::Tensor* idx = context.Input<framework::Tensor>("parent_idx");
    const framework::Tensor* ids =context.Input<framework::Tensor>("ids");
    s->search_start(idx, ids);

    framework::Tensor* out = context.Output<framework::Tensor>("Out");

    framework::TensorCopy(*ids, ids->place(), out);
  }
};

template <typename T>
class TrieSearchWaitCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    VLOG(3) << "TrieSearchWaitCUDAKernel";

    auto s = framework::TrieManager::GetInstance();
    auto place = context.GetPlace();
    auto gpu_ctx = dynamic_cast<phi::GPUContext*>(
                  platform::DeviceContextPool::Instance().Get(place));

    const framework::Tensor* mask = context.Input<framework::Tensor>("X"); // b,s,V
    framework::Tensor* out = context.Output<framework::Tensor>("Out"); // b,s,V
    s->search_wait(); // b,s,x

    auto dims = mask->dims();
    PADDLE_ENFORCE_EQ(
        dims.size(), 2,
        platform::errors::InvalidArgument(
            "Input X'dim size should be equal to 2. "
            "But received X's shape = [%s].",
            dims));

    const int C = dims[0];
    const int N = dims[1];

    int64_t* idx = s->next_out_d_.data<int64_t>();
    int64_t* lod = s->next_lod_d_.data<int64_t>(); // b,s,x

    VLOG(3) << "mask" << framework::PrintTensor<float>(*mask, 100);
    VLOG(3) << "idx" << framework::PrintTensor<int64_t>(s->next_out_d_);
    VLOG(3) << "lod" << framework::PrintTensor<int64_t>(s->next_lod_d_);

    framework::TensorCopy(*mask, mask->place(), out);
    int64_t len = s->next_out_d_.numel();
    float* arr = out->data<float>();

    if (s->next_lod_d_.numel() == 2) {
      MaskKernel1D<<<CUDA_BLOCK(len*C), gpu_ctx->stream()>>>(len*C, len, idx, C, N, arr);
    } else {
      PADDLE_ENFORCE_EQ(
          C+1,
          s->next_lod_d_.numel(),
          platform::errors::InvalidArgument("C+1 != lod %s %s", C+1, s->next_lod_d_.numel()));
      MaskKernel<<<CUDA_BLOCK(len), gpu_ctx->stream()>>>(len, lod, idx, C, N, arr);
    }

    gpu_ctx->Wait();
    VLOG(3) << "out" << framework::PrintTensor<float>(*out, -1);
    // std::cout << "out " << framework::PrintTensor<float>(*out, -1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(trie_search_start,
                        ops::TrieSearchStartCUDAKernel<int64_t>)
REGISTER_OP_CUDA_KERNEL(trie_search_wait,
                        ops::TrieSearchWaitCUDAKernel<float>)