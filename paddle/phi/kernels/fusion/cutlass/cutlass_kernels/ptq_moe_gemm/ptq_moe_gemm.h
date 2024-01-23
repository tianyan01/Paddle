/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm_configs.h"
#include "cuda_runtime_api.h"

namespace phi {

template<typename T> /*The type used for scales/bias/compute*/
class PTQMoeGemmRunner {
public:
  PTQMoeGemmRunner();

  void moe_gemm_bias_act(const int8_t* A,
                         const int8_t* B,
                         const float* weight_scales,
                         const T* biases,
                         T* C,
                         const int64_t* total_rows_before_expert,
                         int64_t total_rows,
                         int64_t gemm_n,
                         int64_t gemm_k,
                         int num_experts,
                         bool do_activation,
                         cudaStream_t stream);

private:
  void dispatch_to_arch(const int8_t* A,
                        const int8_t* B,
                        const float* weight_scales,
                        const T* biases,
                        T* C,
                        const int64_t* total_rows_before_expert,
                        int64_t total_rows,
                        int64_t gemm_n,
                        int64_t gemm_k,
                        int num_experts,
                        bool do_activation,
                        CutlassGemmConfig gemm_config,
                        cudaStream_t stream,
                        int* occupancy = nullptr);

private:
  int sm_;
  int multi_processor_count_;
};

}  // namespace phi