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

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/moe_gemm_kernels_template.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic pop

namespace phi {

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
                        int* occupancy) {
  FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  switch (gemm_config.stages) {
    case 2:
      using DispatcherStages2 = moe_dispatch_stages<T,
                                                    WeightType,
                                                    arch,
                                                    EpilogueTag,
                                                    ThreadblockShape,
                                                    WarpShape,
                                                    2>;
      DispatcherStages2::dispatch(A,
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
      break;
    case 3:
      using DispatcherStages3 = moe_dispatch_stages<T,
                                                    WeightType,
                                                    arch,
                                                    EpilogueTag,
                                                    ThreadblockShape,
                                                    WarpShape,
                                                    3>;
      DispatcherStages3::dispatch(A,
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
      break;
    case 4:
      using DispatcherStages4 = moe_dispatch_stages<T,
                                                    WeightType,
                                                    arch,
                                                    EpilogueTag,
                                                    ThreadblockShape,
                                                    WarpShape,
                                                    4>;
      DispatcherStages4::dispatch(A,
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
      break;
    default:
      std::string err_msg = "dispatch_gemm_config does not support stages " +
                            std::to_string(gemm_config.stages);
      throw std::runtime_error("[FT Error][MoE][dispatch_gemm_config] " +
                               err_msg);
      break;
  }
}
// This overload will handle simt gemms. It is disabled via SFINAE for tensorop.
template <
    typename T,
    typename WeightType,
    typename arch,
    typename EpilogueTag,
    typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr>
void dispatchMoeGemmToCutlass(const T* A,
                              const WeightType* B,
                              const T* weight_scales,
                              const T* biases,
                              T* C,
                              int64_t* total_rows_before_expert,
                              int64_t total_rows,
                              int64_t gemm_n,
                              int64_t gemm_k,
                              int num_experts,
                              CutlassGemmConfig gemm_config,
                              int sm_version,
                              int multi_processor_count,
                              cudaStream_t stream,
                              int* occupancy) {
  switch (gemm_config.tile_config) {
    case CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
      dispatchGemmConfig<T,
                         WeightType,
                         arch,
                         EpilogueTag,
                         cutlass::gemm::GemmShape<128, 128, 8>,
                         cutlass::gemm::GemmShape<64, 64, 8>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          total_rows,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::Undefined:
      throw std::runtime_error("GEMM config undefined.");
      break;
    case CutlassTileConfig::ChooseWithHeuristic:
      throw std::runtime_error("GEMM config should have already been set by heuristic.");
      break;
    default:
      throw std::runtime_error("Unsupported config for float MoE gemm.");
      break;
  }
}
// Tensorop GEMM overload
// Overload for quantize MoE GEMMs. We disable some warp configs here since they
// will not be used and we can improve compile time
template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename std::enable_if<!std::is_same<T, float>::value &&
                                  std::is_same<T, WeightType>::value>::type* =
              nullptr>
void dispatchMoeGemmToCutlass(const T* A,
                              const WeightType* B,
                              const T* weight_scales,
                              const T* biases,
                              T* C,
                              int64_t* total_rows_before_expert,
                              int64_t total_rows,
                              int64_t gemm_n,
                              int64_t gemm_k,
                              int num_experts,
                              CutlassGemmConfig gemm_config,
                              int sm_version,
                              int multi_processor_count,
                              cudaStream_t stream,
                              int* occupancy) {
  switch (gemm_config.tile_config) {
    case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      dispatchGemmConfig<T,
                         WeightType,
                         arch,
                         EpilogueTag,
                         cutlass::gemm::GemmShape<32, 128, 64>,
                         cutlass::gemm::GemmShape<32, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          total_rows,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
      dispatchGemmConfig<T,
                         WeightType,
                         arch,
                         EpilogueTag,
                         cutlass::gemm::GemmShape<64, 128, 64>,
                         cutlass::gemm::GemmShape<32, 64, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          total_rows,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
      dispatchGemmConfig<T,
                         WeightType,
                         arch,
                         EpilogueTag,
                         cutlass::gemm::GemmShape<128, 128, 64>,
                         cutlass::gemm::GemmShape<64, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          total_rows,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::Undefined:
      throw std::runtime_error("GEMM config undefined.");
      break;
    case CutlassTileConfig::ChooseWithHeuristic:
      throw std::runtime_error(
          "GEMM config should have already been set by heuristic.");
      break;
    default:
      throw std::runtime_error(
          "Config is invalid for same type tensorop GEMM.");
      break;
  }
}
// Tensorop GEMM overload
// Overload for quantize MoE GEMMs. We disable some warp configs here since they
// will not be used and we can improve compile time
template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename std::enable_if<!std::is_same<T, float>::value &&
                                  !std::is_same<T, WeightType>::value>::type* =
              nullptr>
void dispatchMoeGemmToCutlass(const T* A,
                              const WeightType* B,
                              const T* weight_scales,
                              const T* biases,
                              T* C,
                              int64_t* total_rows_before_expert,
                              int64_t total_rows,
                              int64_t gemm_n,
                              int64_t gemm_k,
                              int num_experts,
                              CutlassGemmConfig gemm_config,
                              int sm_version,
                              int multi_processor_count,
                              cudaStream_t stream,
                              int* occupancy) {
  switch (gemm_config.tile_config) {
    case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      dispatchGemmConfig<T,
                         WeightType,
                         arch,
                         EpilogueTag,
                         cutlass::gemm::GemmShape<32, 128, 64>,
                         cutlass::gemm::GemmShape<32, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          total_rows,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
      dispatchGemmConfig<T,
                         WeightType,
                         arch,
                         EpilogueTag,
                         cutlass::gemm::GemmShape<64, 128, 64>,
                         cutlass::gemm::GemmShape<64, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          total_rows,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
      dispatchGemmConfig<T,
                         WeightType,
                         arch,
                         EpilogueTag,
                         cutlass::gemm::GemmShape<128, 128, 64>,
                         cutlass::gemm::GemmShape<128, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          total_rows_before_expert,
          total_rows,
          gemm_n,
          gemm_k,
          num_experts,
          gemm_config,
          multi_processor_count,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::Undefined:
      throw std::runtime_error("GEMM config undefined.");
      break;
    case CutlassTileConfig::ChooseWithHeuristic:
      throw std::runtime_error(
          "GEMM config should have already been set by heuristic.");
      break;
    default:
      throw std::runtime_error(
          "Config is invalid for mixed type tensorop GEMM.");
      break;
  }
}

template <typename T, typename WeightType>
MoeGemmRunner<T, WeightType>::MoeGemmRunner() {
  FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  int device{-1};
  check_cuda_error(cudaGetDevice(&device));
  sm_ = getSMVersion();
  check_cuda_error(cudaDeviceGetAttribute(
      &multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename T, typename WeightType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType>::dispatch_to_arch<EpilogueTag>(
    const T* A,
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
    cudaStream_t stream,
    int* occupancy) {
  FT_LOG_DEBUG(__PRETTY_FUNCTION__);

  if (sm_ >= 70 && sm_ < 75) {
#if defined(USE_FPAINTB_GEMM_WITH_SM70)
    dispatchMoeGemmToCutlass<T, WeightType, cutlass::arch::Sm70, EpilogueTag>(
        A,
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
        sm_,
        multi_processor_count_,
        stream,
        occupancy);
#else
    throw std::runtime_error(
        "[MoeGemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS mixed "
        "type GEMM");
#endif
  }
#if defined(USE_FPAINTB_GEMM_WITH_SM75)
  else if (sm_ >= 75 && sm_ < 80) {
    dispatchMoeGemmToCutlass<T, WeightType, cutlass::arch::Sm75, EpilogueTag>(
        A,
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
        sm_,
        multi_processor_count_,
        stream,
        occupancy);
  }
#endif
#if defined(USE_FPAINTB_GEMM_WITH_SM80) || defined(USE_FPAINTB_GEMM_WITH_SM90)
  else if (sm_ >= 80 && sm_ <= 90) {
    dispatchMoeGemmToCutlass<T, WeightType, cutlass::arch::Sm80, EpilogueTag>(
        A,
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
        sm_,
        multi_processor_count_,
        stream,
        occupancy);
  }
#endif
  else {
    throw std::runtime_error(
        "[FT Error][MoE][GEMM Dispatch] Arch unsupported for MoE GEMM");
  }
}

template <typename T, typename WeightType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType>::run_gemm<EpilogueTag>(
    const T* A,
    const WeightType* B,
    const T* weight_scales,
    const T* biases,
    T* C,
    int64_t* total_rows_before_expert,
    int64_t total_rows,
    int64_t gemm_n,
    int64_t gemm_k,
    int num_experts,
    cudaStream_t stream) {
  FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  int64_t key =
      static_cast<int64_t>((static_cast<int64_t>(num_experts) << 44) |
                           static_cast<int64_t>(gemm_n) << 22 | gemm_k);
  CutlassGemmConfig chosen_config;
  auto it = config_cache_.find(key);
  if (it == config_cache_.end()) {
    static constexpr bool is_weight_only = !std::is_same<T, WeightType>::value;
    static constexpr bool only_simt_configs = std::is_same<T, float>::value;

    static constexpr int workspace_bytes = 0;  // No workspace for MoE GEMMs.
    static constexpr int split_k_limit =
        1;  // MoE GEMM does not support split-k.
    std::vector<CutlassGemmConfig> candidate_configs = get_candidate_configs(
        sm_, is_weight_only, only_simt_configs, false, split_k_limit);
    std::vector<int> occupancies(candidate_configs.size());

    for (size_t ii = 0; ii < candidate_configs.size(); ++ii) {
      dispatch_to_arch<EpilogueTag>(A,
                                    B,
                                    weight_scales,
                                    biases,
                                    C,
                                    total_rows_before_expert,
                                    total_rows,
                                    gemm_n,
                                    gemm_k,
                                    num_experts,
                                    candidate_configs[ii],
                                    stream,
                                    &occupancies[ii]);
    }

    chosen_config =
        estimate_best_config_from_occupancies(candidate_configs,
                                              occupancies,
                                              total_rows,
                                              gemm_n,
                                              gemm_k,
                                              num_experts,
                                              split_k_limit,
                                              workspace_bytes,
                                              multi_processor_count_,
                                              is_weight_only);
  } else {
    chosen_config = it->second;
  }

  dispatch_to_arch<EpilogueTag>(A,
                                B,
                                weight_scales,
                                biases,
                                C,
                                total_rows_before_expert,
                                total_rows,
                                gemm_n,
                                gemm_k,
                                num_experts,
                                chosen_config,
                                stream);
}

template <typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::moe_gemm_bias_act(
    const T* A,
    const WeightType* B,
    const T* weight_scales,
    const T* biases,
    T* C,
    int64_t* total_rows_before_expert,
    int64_t total_rows,
    int64_t gemm_n,
    int64_t gemm_k,
    int num_experts,
    ActivationType activation_type,
    cudaStream_t stream) {
  FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  switch (activation_type) {
    case ActivationType::Relu:
      run_gemm<EpilogueOpBiasReLU>(A,
                                   B,
                                   weight_scales,
                                   biases,
                                   C,
                                   total_rows_before_expert,
                                   total_rows,
                                   gemm_n,
                                   gemm_k,
                                   num_experts,
                                   stream);
      break;
    case ActivationType::Gelu:
      run_gemm<EpilogueOpBiasFtGelu>(A,
                                     B,
                                     weight_scales,
                                     biases,
                                     C,
                                     total_rows_before_expert,
                                     total_rows,
                                     gemm_n,
                                     gemm_k,
                                     num_experts,
                                     stream);
      break;
    /**
    case ActivationType::Silu:
      run_gemm<EpilogueOpBiasSilu>(A,
                                   B,
                                   weight_scales,
                                   biases,
                                   C,
                                   total_rows_before_expert,
                                   total_rows,
                                   gemm_n,
                                   gemm_k,
                                   num_experts,
                                   stream);
      break;
    */
    case ActivationType::Identity:
      run_gemm<EpilogueOpBias>(A,
                               B,
                               weight_scales,
                               biases,
                               C,
                               total_rows_before_expert,
                               total_rows,
                               gemm_n,
                               gemm_k,
                               num_experts,
                               stream);
      break;
    case ActivationType::InvalidType:
      FT_CHECK_WITH_INFO(false, "Activation type for fpA_intB must be valid.");
      break;
    default: {
      if (isGatedActivation(activation_type)) {
        FT_CHECK_WITH_INFO(false, "Fused gated activations not supported");
      } else {
        FT_CHECK_WITH_INFO(false, "Invalid activation type.");
      }
    }
  }
}

template <typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::moe_gemm(const T* A,
                                            const WeightType* B,
                                            const T* weight_scales,
                                            T* C,
                                            int64_t* total_rows_before_expert,
                                            int64_t total_rows,
                                            int64_t gemm_n,
                                            int64_t gemm_k,
                                            int num_experts,
                                            cudaStream_t stream) {
  FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  run_gemm<EpilogueOpDefault>(A,
                              B,
                              weight_scales,
                              nullptr,
                              C,
                              total_rows_before_expert,
                              total_rows,
                              gemm_n,
                              gemm_k,
                              num_experts,
                              stream);
}

template class MoeGemmRunner<half, uint8_t>;
template class MoeGemmRunner<half, half>;
template class MoeGemmRunner<half, cutlass::uint4b_t>;
template class MoeGemmRunner<float, float>;
#ifdef PADDLE_CUDA_BF16
template class MoeGemmRunner<__nv_bfloat16, uint8_t>;
template class MoeGemmRunner<__nv_bfloat16, cutlass::uint4b_t>;
#endif
}  // namespace phi
