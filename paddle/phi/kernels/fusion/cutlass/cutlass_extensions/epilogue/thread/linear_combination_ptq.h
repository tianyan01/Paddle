/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing linear combination operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

template<
  typename ElementCompute_,
  typename ElementOutput_,
  bool IsHeavy = false,
  template<typename T> class ActivationFunctor = cutlass::epilogue::thread::GELU_taylor
>
class LinearCombinationPTQ {
public:
  using ElementCompute = ElementCompute_;
  using ElementOutput = ElementOutput_;
  using ElementSource = ElementOutput; // fp16, bias
  using ElementAccumulator = int32_t; // int32

  static int const kCount = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  static const ScaleType::Kind kScale = ScaleType::OnlyAlphaPerChannelScaling;
  // static constexpr bool IsPerChannelScalingSupported = true;
  static bool const kIsHeavy = IsHeavy;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>; // int32
  using FragmentCompute = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = FloatRoundStyle::round_to_nearest;

  /// Host-constructable parameters structure
  struct Params
  {
    float beta;                   ///< scales source tensor
    bool do_act;                 ///< if true, apply activation function

    CUTLASS_HOST_DEVICE
    Params():
      beta(float(0)),
      do_act(false) { }

    CUTLASS_HOST_DEVICE
    Params(float _beta, bool _do_act):
      beta(_beta),
      do_act(_do_act) { }
  };

  // add new fun
  // struct fp32multiply_fp16add {
  //   using A = FragmentMul;
  //   using B = A;
  //   using C = FragmentOutput;
  //   NumericArrayConverter<cutlass::half_t, float, kCount, kRound> converter;
  //   CUTLASS_HOST_DEVICE
  //   C operator()(A const &a, B const &b, C const &c) const {
  //     C res = converter(a * b);
  //     return res + c;
  //   }
  // };

private:

  //
  // Data members
  //
  float beta_ = float(0);
  bool do_act_ = false;

public:

  /// Constructs the function object
  CUTLASS_HOST_DEVICE
  LinearCombinationPTQ(Params const& params) {
    beta_ = params.beta;
    do_act_ = params.do_act;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return true; // always need source
  }

  CUTLASS_HOST_DEVICE
  bool is_beta_vector() const {
    return false; // beta always not a vector
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
  }

  /// Computes linear scaling with source: D = act(scale * accumulator + bias)
  /// scalar_beta is next in scale
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const& accumulator, // A * B, int32
      FragmentCompute const& scale, // scale, float
      FragmentSource const& bias, // fp16
      bool is_print_debug) const {
    // convert accum from int32 to fp32
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, kRound> accumulator_converter;
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);
    FragmentOutput intermediate; // fp16
    multiplies<FragmentCompute> mul; // dequant
    plus<FragmentSource> add;  // add bias
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, kRound> out_converter;
    intermediate = add(out_converter(mul(scale, converted_accumulator)), bias);

    ActivationFunctor<FragmentOutput> activation;
    intermediate = do_act_ ? activation(intermediate) : intermediate;
    return intermediate;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////