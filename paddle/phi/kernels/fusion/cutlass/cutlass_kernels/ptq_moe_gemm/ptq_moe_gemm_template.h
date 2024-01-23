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

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/gemm/device/gemm_grouped.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/kernel/default_moe_gemm_grouped.h"

#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/compute_occupancy.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/kernel/ptq_moe_kernel.h"

#pragma GCC diagnostic pop

#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/cutlass_heuristic.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/ptq_moe_gemm/ptq_moe_gemm.h"
#include "paddle/phi/kernels/fusion/cutlass/utils/cuda_utils.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>

namespace phi {

// ============================= Variable batched Gemm things ===========================
template<typename T,
         typename arch,
         typename ThreadblockShape,
         typename WarpShape,
         int Stages>
void generic_moe_gemm_kernelLauncher(const int8_t* A,
                                     const int8_t* B,
                                     const float* weight_scales,
                                     const T* biases,
                                     T* C,
                                     const int64_t* total_rows_before_expert,
                                     int64_t gemm_n,
                                     int64_t gemm_k,
                                     int num_experts,
                                     bool do_activation,
                                     CutlassGemmConfig gemm_config,
                                     const int multi_processor_count,
                                     cudaStream_t stream,
                                     int* kernel_occupancy = nullptr) {
  if (gemm_config.split_k_style != SplitKStyle::NO_SPLIT_K) {
    throw std::runtime_error("[FT Error][MoeGemm] Grouped gemm does not support split-k");
  }
  using InputType = int8_t;
  using ElementAccumulator = int32_t;
  using ComputeType = float;
  using OutputType =
      typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;

  using EpilogueOutputOp = 
      cutlass::epilogue::thread::LinearCombinationPTQ<ComputeType,
                                                      OutputType,
                                                      false, // is_heavy
                                                      cutlass::epilogue::thread::GELU_taylor>;

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  // Finally, set up the kernel.
  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultMoeGemmGrouped<
      InputType,
      cutlass::layout::RowMajor,
      cutlass::ComplexTransform::kNone,
      16, // 128 / 8
      InputType,
      cutlass::layout::ColumnMajor,
      cutlass::ComplexTransform::kNone,
      16, // 128 / 8
      OutputType, // output type
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      arch,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      Stages,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
      cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;

  using GemmKernel = cutlass::gemm::kernel::PTQMoeGemm<typename GemmKernel_::Mma,
                                                       typename GemmKernel_::Epilogue,
                                                       typename GemmKernel_::ThreadblockSwizzle,
                                                       arch,  // Ensure top level arch is used for dispatch
                                                       GemmKernel_::kGroupScheduleMode>;

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  if (kernel_occupancy != nullptr) {
      *kernel_occupancy = compute_occupancy_for_kernel<GemmKernel>();
      return;
  }
  int occupancy = std::min(2, GemmGrouped::maximum_active_blocks());
  if (occupancy == 0) {
      throw std::runtime_error(
          "[FT Error][MoE Runner] GPU lacks the shared memory resources to run GroupedGEMM kernel");
  }
  const int threadblock_count = multi_processor_count * occupancy;

  typename EpilogueOutputOp::Params epilogue_op(ComputeType(0.f), do_activation);

  typename GemmGrouped::Arguments args(num_experts,
                                       threadblock_count,
                                       epilogue_op,
                                       A,
                                       B,
                                       weight_scales,
                                       reinterpret_cast<const OutputType*>(biases),
                                       reinterpret_cast<OutputType*>(C),
                                       total_rows_before_expert,
                                       gemm_n,
                                       gemm_k);

  GemmGrouped gemm;

  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg =
        "MoEFC kernel will fail for params. Error: " + std::string(cutlassGetStatusString(can_implement));
    throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
  }

  auto init_status = gemm.initialize(args);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to initialize cutlass variable batched gemm. Error: "
                          + std::string(cutlassGetStatusString(init_status));
    throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
  }

  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "Failed to run cutlass variable batched gemm. Error: " + std::string(cutlassGetStatusString(run_status));
    throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
  }
}

template<typename T,
         typename arch,
         typename ThreadblockShape,
         typename WarpShape,
         int Stages,
         typename Enable = void>
struct dispatch_stages {
  static void dispatch(const int8_t* A,
                        const int8_t* B,
                        const float* weight_scales,
                        const T* biases,
                        T* C,
                        const int64_t* total_rows_before_expert,
                        int64_t gemm_n,
                        int64_t gemm_k,
                        int num_experts,
                        bool do_activation,
                        CutlassGemmConfig gemm_config,
                        int multi_processor_count,
                        cudaStream_t stream,
                        int* occupancy = nullptr) {
    std::string err_msg = "Cutlass moe gemm. Not instantiates for arch "
                          + std::to_string(arch::kMinComputeCapability) + " with stages set to "
                          + std::to_string(Stages);
    throw std::runtime_error("[FT Error][dispatch_stages::dispatch] " + err_msg);
  }
};

template<typename T,
         typename arch,
         typename ThreadblockShape,
         typename WarpShape>
struct dispatch_stages<T, arch, ThreadblockShape, WarpShape, 2> {
  static void dispatch(const int8_t* A,
                        const int8_t* B,
                        const float* weight_scales,
                        const T* biases,
                        T* C,
                        const int64_t* total_rows_before_expert,
                        int64_t gemm_n,
                        int64_t gemm_k,
                        int num_experts,
                        bool do_activation,
                        CutlassGemmConfig gemm_config,
                        int multi_processor_count,
                        cudaStream_t stream,
                        int* occupancy = nullptr) {
    generic_moe_gemm_kernelLauncher<T, arch, ThreadblockShape, WarpShape, 2>(
        A,
        B,
        weight_scales,
        biases,
        C,
        total_rows_before_expert,
        gemm_n,
        gemm_k,
        num_experts,
        do_activation,
        gemm_config,
        multi_processor_count,
        stream,
        occupancy);
  }
};

template<typename T,
         typename ThreadblockShape,
         typename WarpShape,
         int Stages>
struct dispatch_stages<T,
                       cutlass::arch::Sm80,
                       ThreadblockShape,
                       WarpShape,
                       Stages,
                       typename std::enable_if<(Stages > 2)>::type> {
  static void dispatch(const int8_t* A,
                        const int8_t* B,
                        const float* weight_scales,
                        const T* biases,
                        T* C,
                        const int64_t* total_rows_before_expert,
                        int64_t gemm_n,
                        int64_t gemm_k,
                        int num_experts,
                        bool do_activation,
                        CutlassGemmConfig gemm_config,
                        int multi_processor_count,
                        cudaStream_t stream,
                        int* occupancy = nullptr) {
    generic_moe_gemm_kernelLauncher<T,
                                    cutlass::arch::Sm80,
                                    ThreadblockShape,
                                    WarpShape,
                                    Stages>(A,
                                            B,
                                            weight_scales,
                                            biases,
                                            C,
                                            total_rows_before_expert,
                                            gemm_n,
                                            gemm_k,
                                            num_experts,
                                            do_activation,
                                            gemm_config,
                                            multi_processor_count,
                                            stream,
                                            occupancy);
  }
};

template<typename T,
         typename arch,
         typename ThreadblockShape,
         typename WarpShape>
void dispatch_gemm_config(const int8_t* A,
                          const int8_t* B,
                          const float* weight_scales,
                          const T* biases,
                          T* C,
                          const int64_t* total_rows_before_expert,
                          int64_t gemm_n,
                          int64_t gemm_k,
                          int num_experts,
                          bool do_activation,
                          CutlassGemmConfig gemm_config,
                          int multi_processor_count,
                          cudaStream_t stream,
                          int* occupancy = nullptr) {
  switch (gemm_config.stages) {
    case 2:
      using DispatcherStages2 = dispatch_stages<T, arch, ThreadblockShape, WarpShape, 2>;
      DispatcherStages2::dispatch(A,
                                  B,
                                  weight_scales,
                                  biases,
                                  C,
                                  total_rows_before_expert,
                                  gemm_n,
                                  gemm_k,
                                  num_experts,
                                  do_activation,
                                  gemm_config,
                                  multi_processor_count,
                                  stream,
                                  occupancy);
      break;
    case 3:
      using DispatcherStages3 = dispatch_stages<T, arch, ThreadblockShape, WarpShape, 3>;
      DispatcherStages3::dispatch(A,
                                  B,
                                  weight_scales,
                                  biases,
                                  C,
                                  total_rows_before_expert,
                                  gemm_n,
                                  gemm_k,
                                  num_experts,
                                  do_activation,
                                  gemm_config,
                                  multi_processor_count,
                                  stream,
                                  occupancy);
      break;
    case 4:
      using DispatcherStages4 = dispatch_stages<T, arch, ThreadblockShape, WarpShape, 4>;
      DispatcherStages4::dispatch(A,
                                  B,
                                  weight_scales,
                                  biases,
                                  C,
                                  total_rows_before_expert,
                                  gemm_n,
                                  gemm_k,
                                  num_experts,
                                  do_activation,
                                  gemm_config,
                                  multi_processor_count,
                                  stream,
                                  occupancy);
      break;
    default:
      std::string err_msg = "dispatch_gemm_config does not support stages " + std::to_string(gemm_config.stages);
      throw std::runtime_error("[FT Error][MoE][dispatch_gemm_config] " + err_msg);
      break;
  }
}

template<typename T,
         typename arch>
void dispatch_moe_gemm_to_cutlass(const int8_t* A,
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
                                  int sm_version,
                                  int multi_processor_count,
                                  cudaStream_t stream,
                                  int* occupancy = nullptr) {
  switch (gemm_config.tile_config) {
    case CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
      dispatch_gemm_config<T,
                          arch,
                          cutlass::gemm::GemmShape<128, 256, 64>,
                          cutlass::gemm::GemmShape<64, 64, 64>>(A,
                                                                B,
                                                                weight_scales,
                                                                biases,
                                                                C,
                                                                total_rows_before_expert,
                                                                gemm_n,
                                                                gemm_k,
                                                                num_experts,
                                                                do_activation,
                                                                gemm_config,
                                                                multi_processor_count,
                                                                stream,
                                                                occupancy);
      break;
    case CutlassTileConfig::Undefined:
      throw std::runtime_error("[FT Error][dispatch_moe_gemm_to_cutlass] gemm config undefined.");
      break;
    case CutlassTileConfig::ChooseWithHeuristic:
      throw std::runtime_error(
          "[FT Error][dispatch_moe_gemm_to_cutlass] gemm config should have already been set by heuristic.");
      break;
    default:
      throw std::runtime_error(
          "[FT Error][dispatch_moe_gemm_to_cutlass] Config is invalid for same type MoE tensorop GEMM.");
      break;
  }
}

template<typename T>
PTQMoeGemmRunner<T>::PTQMoeGemmRunner() {
  int device{-1};
  check_cuda_error(cudaGetDevice(&device));
  sm_ = getSMVersion();
  check_cuda_error(cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template<typename T>
void PTQMoeGemmRunner<T>::dispatch_to_arch(const int8_t* A,
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
                                           int* occupancy) {
  if (sm_ >= 80 && sm_ <= 90) {
    dispatch_moe_gemm_to_cutlass<T, cutlass::arch::Sm80>(A,
                                                         B,
                                                         weight_scales,
                                                         biases,
                                                         C,
                                                         total_rows_before_expert,
                                                         total_rows,
                                                         gemm_n,
                                                         gemm_k,
                                                         num_experts,
                                                         do_activation,
                                                         gemm_config,
                                                         sm_,
                                                         multi_processor_count_,
                                                         stream,
                                                         occupancy);
  } else {
    throw std::runtime_error("[FT Error][MoE][GEMM Dispatch] Arch unsupported for MoE GEMM");
  }
}

template<typename T>
void PTQMoeGemmRunner<T>::moe_gemm_bias_act(const int8_t* A,
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
                                            cudaStream_t stream) {
  static constexpr int workspace_bytes = 0;  // No workspace for MoE GEMMs.
  static constexpr int split_k_limit = 1;  // MoE GEMM does not support split-k.
  std::vector<CutlassGemmConfig> candidate_configs = get_candidate_configs(sm_, CutlassGemmType::PTQ, split_k_limit);
  std::vector<int> occupancies(candidate_configs.size());

  for (size_t ii = 0; ii < candidate_configs.size(); ++ii) {
    dispatch_to_arch(A,
                     B,
                     weight_scales,
                     biases,
                     C,
                     total_rows_before_expert,
                     total_rows,
                     gemm_n,
                     gemm_k,
                     num_experts,
                     do_activation,
                     candidate_configs[ii],
                     stream,
                     &occupancies[ii]);
  }

  CutlassGemmConfig chosen_config = estimate_best_config_from_occupancies(candidate_configs,
                                                                          occupancies,
                                                                          total_rows,
                                                                          gemm_n,
                                                                          gemm_k,
                                                                          num_experts,
                                                                          split_k_limit,
                                                                          workspace_bytes,
                                                                          multi_processor_count_,
                                                                          false);

  dispatch_to_arch(A,
                   B,
                   weight_scales,
                   biases,
                   C,
                   total_rows_before_expert,
                   total_rows,
                   gemm_n,
                   gemm_k,
                   num_experts,
                   do_activation,
                   chosen_config,
                   stream);
}

}  // namespace phi