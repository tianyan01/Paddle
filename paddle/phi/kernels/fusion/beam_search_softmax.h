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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void BeamSearchSoftmaxKernel(const Context &dev_ctx,
                             const DenseTensor &logits,
                             const DenseTensor &cum_scores,
                             const DenseTensor &sequence_lengths,
                             const DenseTensor &stop_flags,
                             const DenseTensor &end_ids,
                             const DenseTensor &step_ids,
                             const DenseTensor &last_cache_ids,
                             const DenseTensor &last_beam_offsets,
                             int beam_size,
                             int max_seq_len,
                             int max_dec_len,
                             bool fuse_softmax,
                             bool early_stop,
                             DenseTensor *ids_this_time,
                             DenseTensor *out_cum_scores,
                             DenseTensor *cache_ids,
                             DenseTensor *beam_offsets,
                             DenseTensor *parent_idx,
                             DenseTensor *stop_flags_out,
                             DenseTensor *seq_lens_out,
                             DenseTensor *step_ids_out);

}  // namespace fusion
}  // namespace phi
