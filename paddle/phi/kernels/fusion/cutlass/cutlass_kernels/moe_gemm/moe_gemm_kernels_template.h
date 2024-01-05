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
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/compute_occupancy.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/epilogue_helpers.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/threadblock/default_mma.h"

#pragma GCC diagnostic pop

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/activation_types.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/cutlass_heuristic.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/autogen/arch_define.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"

namespace phi {

// ============================= Variable batched Gemm things
// ===========================
template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages>
void generic_moe_gemm_kernelLauncher(const T* A,
                                     const WeightType* B,
                                     const T* weight_scales,
                                     const T* biases,
                                     T* C,
                                     int64_t* total_rows_before_expert,
                                     int64_t num_rows,
                                     int64_t gemm_n,
                                     int64_t gemm_k,
                                     int num_experts,
                                     CutlassGemmConfig gemm_config,
                                     const int multi_processor_count,
                                     cudaStream_t stream,
                                     int* kernel_occupancy) {
  FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  if (gemm_config.split_k_style != SplitKStyle::NO_SPLIT_K) {
    throw std::runtime_error(
        "[FT Error][MoeGemm] Grouped gemm does not support split-k");
  }

#ifdef PADDLE_CUDA_BF16
  static_assert(cutlass::platform::is_same<T, __nv_bfloat16>::value ||
                    cutlass::platform::is_same<T, half>::value ||
                    cutlass::platform::is_same<T, float>::value,
                "Specialized for bfloat16, half, float");
#else
  static_assert(cutlass::platform::is_same<T, half>::value ||
                    cutlass::platform::is_same<T, float>::value,
                "Specialized for half, float");
#endif

  static_assert(
      cutlass::platform::is_same<T, WeightType>::value ||
          cutlass::platform::is_same<WeightType, uint8_t>::value ||
          cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value,
      "");

  // The cutlass type for the input elements. This is needed to convert to
  // cutlass::half_t if necessary.
  using ElementType_ = typename cutlass::platform::conditional<
      cutlass::platform::is_same<T, half>::value,
      cutlass::half_t,
      T>::type;
#ifdef PADDLE_CUDA_BF16
  using ElementType = typename cutlass::platform::conditional<
      cutlass::platform::is_same<ElementType_, __nv_bfloat16>::value,
      cutlass::bfloat16_t,
      ElementType_>::type;
#else
  using ElementType = ElementType_;
#endif

  using CutlassWeightType_ = typename cutlass::platform::conditional<
      cutlass::platform::is_same<WeightType, half>::value,
      cutlass::half_t,
      WeightType>::type;
#ifdef PADDLE_CUDA_BF16
  using CutlassWeightType = typename cutlass::platform::conditional<
      cutlass::platform::is_same<CutlassWeightType_, __nv_bfloat16>::value,
      cutlass::bfloat16_t,
      CutlassWeightType_>::type;
#else
  using CutlassWeightType = CutlassWeightType_;
#endif

  // We need separate config for each architecture since we will target
  // different tensorcore instructions. For float, we do not target TCs.
  using MixedGemmArchTraits = cutlass::gemm::kernel::
      MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
  using ElementAccumulator = typename MixedGemmArchTraits::AccType;

  using EpilogueOp = typename Epilogue<ElementType,
                                       MixedGemmArchTraits::ElementsPerAccessC,
                                       ElementAccumulator,
                                       EpilogueTag>::Op;

  // Finally, set up the kernel.
  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      ElementType,
      cutlass::layout::RowMajor,
      cutlass::ComplexTransform::kNone,
      MixedGemmArchTraits::ElementsPerAccessA,
      CutlassWeightType,
      typename MixedGemmArchTraits::LayoutB,
      cutlass::ComplexTransform::kNone,
      MixedGemmArchTraits::ElementsPerAccessB,
      ElementType,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      typename MixedGemmArchTraits::OperatorClass,
      arch,
      ThreadblockShape,
      WarpShape,
      typename MixedGemmArchTraits::InstructionShape,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      Stages,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
      typename MixedGemmArchTraits::Operator>::GemmKernel;

  using GemmKernel =
      cutlass::gemm::kernel::MoeFCGemm<typename GemmKernel_::Mma,
                                       typename GemmKernel_::Epilogue,
                                       typename GemmKernel_::ThreadblockSwizzle,
                                       arch,  // Ensure top level arch is used
                                              // for dispatch
                                       GemmKernel_::kGroupScheduleMode>;

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  if (kernel_occupancy != nullptr) {
    *kernel_occupancy = compute_occupancy_for_kernel<GemmKernel>();
    return;
  }
  int occupancy = std::min(2, GemmGrouped::maximum_active_blocks());
  if (occupancy == 0) {
    throw std::runtime_error(
        "[FT Error][MoE Runner] GPU lacks the shared memory resources to run "
        "GroupedGEMM kernel");
  }
  const int threadblock_count = multi_processor_count * occupancy;

  typename EpilogueOp::Params epilogue_op(ElementAccumulator(1.f),
                                          ElementAccumulator(0.f));

  const int group_size = gemm_k;
  typename GemmGrouped::Arguments args(
      num_experts,
      threadblock_count,
      group_size,
      epilogue_op,
      reinterpret_cast<const ElementType*>(A),
      reinterpret_cast<const CutlassWeightType*>(B),
      reinterpret_cast<const ElementType*>(weight_scales),
      reinterpret_cast<const ElementType*>(biases),
      reinterpret_cast<ElementType*>(C),
      total_rows_before_expert,
      gemm_n,
      gemm_k);

  GemmGrouped gemm;

  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg = "MoEFC kernel will fail for params. Error: " +
                          std::string(cutlassGetStatusString(can_implement));
    throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
  }

  auto init_status = gemm.initialize(args);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "Failed to initialize cutlass variable batched gemm. Error: " +
        std::string(cutlassGetStatusString(init_status));
    throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
  }

  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "Failed to run cutlass variable batched gemm. Error: " +
        std::string(cutlassGetStatusString(run_status));
    throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
  }
}

template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages>
void generic_moe_gemm_kernelLauncher_template(const T* A,
                                              const WeightType* B,
                                              const T* weight_scales,
                                              const T* biases,
                                              T* C,
                                              int64_t* total_rows_before_expert,
                                              int64_t num_rows,
                                              int64_t gemm_n,
                                              int64_t gemm_k,
                                              int num_experts,
                                              CutlassGemmConfig gemm_config,
                                              const int multi_processor_count,
                                              cudaStream_t stream,
                                              int* kernel_occupancy);

template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages,
          typename Enable = void>
struct moe_dispatch_stages {
  static void dispatch(const T* A,
                       const WeightType* B,
                       const T* weight_scales,
                       const T* biases,
                       T* C,
                       int64_t* total_rows_before_expert,
                       int64_t num_rows,
                       int64_t gemm_n,
                       int64_t gemm_k,
                       int num_experts,
                       CutlassGemmConfig gemm_config,
                       int multi_processor_count,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    std::string err_msg = "Cutlass fpA_intB gemm. Not instantiates for arch " +
                          std::to_string(arch::kMinComputeCapability) +
                          " with stages set to " + std::to_string(Stages);
    throw std::runtime_error("[FT Error][dispatch_stages::dispatch] " +
                             err_msg);
  }
};

template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape>
struct moe_dispatch_stages<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           ThreadblockShape,
                           WarpShape,
                           2> {
  static void dispatch(const T* A,
                       const WeightType* B,
                       const T* weight_scales,
                       const T* biases,
                       T* C,
                       int64_t* total_rows_before_expert,
                       int64_t num_rows,
                       int64_t gemm_n,
                       int64_t gemm_k,
                       int num_experts,
                       CutlassGemmConfig gemm_config,
                       int multi_processor_count,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    generic_moe_gemm_kernelLauncher_template<T,
                                             WeightType,
                                             arch,
                                             EpilogueTag,
                                             ThreadblockShape,
                                             WarpShape,
                                             2>(A,
                                                B,
                                                weight_scales,
                                                biases,
                                                C,
                                                total_rows_before_expert,
                                                num_rows,
                                                gemm_n,
                                                gemm_k,
                                                num_experts,
                                                gemm_config,
                                                multi_processor_count,
                                                stream,
                                                occupancy);
  }
};
#if defined(USE_FPAINTB_GEMM_WITH_SM80)
template <typename T,
          typename WeightType,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages>
struct moe_dispatch_stages<T,
                           WeightType,
                           cutlass::arch::Sm80,
                           EpilogueTag,
                           ThreadblockShape,
                           WarpShape,
                           Stages,
                           typename std::enable_if<(Stages > 2)>::type> {
  static void dispatch(const T* A,
                       const WeightType* B,
                       const T* weight_scales,
                       const T* biases,
                       T* C,
                       int64_t* total_rows_before_expert,
                       int64_t num_rows,
                       int64_t gemm_n,
                       int64_t gemm_k,
                       int num_experts,
                       CutlassGemmConfig gemm_config,
                       int multi_processor_count,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    generic_moe_gemm_kernelLauncher_template<T,
                                             WeightType,
                                             cutlass::arch::Sm80,
                                             EpilogueTag,
                                             ThreadblockShape,
                                             WarpShape,
                                             Stages>(A,
                                                     B,
                                                     weight_scales,
                                                     biases,
                                                     C,
                                                     total_rows_before_expert,
                                                     num_rows,
                                                     gemm_n,
                                                     gemm_k,
                                                     num_experts,
                                                     gemm_config,
                                                     multi_processor_count,
                                                     stream,
                                                     occupancy);
  }
};
#endif
template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape>
void dispatchGemmConfig(const T* A,
                        const WeightType* B,
                        const T* weight_scales,
                        const T* biases,
                        T* C,
                        int64_t* total_rows_before_expert,
                        int64_t num_rows,
                        int64_t gemm_n,
                        int64_t gemm_k,
                        int num_experts,
                        CutlassGemmConfig gemm_config,
                        int multi_processor_count,
                        cudaStream_t stream,
                        int* occupancy);

}  // namespace phi
