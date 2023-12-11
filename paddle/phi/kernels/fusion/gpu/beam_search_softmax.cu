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

#include <algorithm>

#include <cub/cub.cuh>
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/fusion/beam_search_softmax.h"

namespace phi {
namespace fusion {

#define FLT_MAX 1e38
// #define DEBUG_BEAM_SEARCH_SOFTMAX

#define CASE_K(K)                                                   \
  case K:                                                           \
    invokeTopKSoftMaxLauncher<T, 2 * K, Context>(dev_ctx,           \
                                                 log_probs,         \
                                                 stop_flags,        \
                                                 sequence_lengths,  \
                                                 cum_log_probs,     \
                                                 step_ids,          \
                                                 last_cache_ids,    \
                                                 last_beam_offsets, \
                                                 end_ids,           \
                                                 out_cum_log_probs, \
                                                 stop_flags_out,    \
                                                 seq_lens_out,      \
                                                 step_ids_out,      \
                                                 ids,               \
                                                 tmp_ids,           \
                                                 tmp_vals,          \
                                                 parent_idx,        \
                                                 cache_ids,         \
                                                 beam_offsets,      \
                                                 batch_size,        \
                                                 beam_size,         \
                                                 vocab_size,        \
                                                 max_seq_len,       \
                                                 max_dec_len,       \
                                                 fuse_softmax,      \
                                                 early_stop,        \
                                                 length_penalty,    \
                                                 stream);           \
    break

struct __align__(8) DySoftMaxStruct {
  float logit;
  float score;
};

__device__ __forceinline__ DySoftMaxStruct
reduce_softmax_op(DySoftMaxStruct a, DySoftMaxStruct b) {
  bool a_bigger = (a.logit > b.logit);
  DySoftMaxStruct bigger_m = a_bigger ? a : b;
  DySoftMaxStruct smaller_m = a_bigger ? b : a;
  DySoftMaxStruct res;
  res.score = bigger_m.score +
              smaller_m.score * exp(smaller_m.logit - bigger_m.logit);
  res.logit = bigger_m.logit;
  return res;
}

template <typename T, int K>
struct TopK {
  int ids[K];
  T vals[K];
  int parent_ids[K];

  __device__ __forceinline__ void insert(T elem, int elem_id) {
    if (elem > vals[K - 1] || (ids[K - 1] == -1) ||
        ((elem == vals[K - 1]) && (elem_id < ids[K - 1]))) {
      vals[K - 1] = elem;
      ids[K - 1] = elem_id;
    }

    for (int k = K - 2; k >= 0; --k) {
      if ((vals[k + 1] > vals[k]) || (ids[k] == -1) ||
          ((vals[k + 1] == vals[k]) && (ids[k + 1] < ids[k]))) {
        T tmp_val = vals[k];
        int tmp_id = ids[k];
        vals[k] = vals[k + 1];
        ids[k] = ids[k + 1];
        vals[k + 1] = tmp_val;
        ids[k + 1] = tmp_id;
      }
    }
  }

  __device__ __forceinline__ void insert(T elem, int elem_id, int parent_id) {
    if (elem > vals[K - 1] || (ids[K - 1] == -1) ||
        ((elem == vals[K - 1]) && (elem_id < ids[K - 1]))) {
      vals[K - 1] = elem;
      ids[K - 1] = elem_id;
      parent_ids[K - 1] = parent_id;
    }

    for (int k = K - 2; k >= 0; --k) {
      if ((vals[k + 1] > vals[k]) || (ids[k] == -1) ||
          ((vals[k + 1] == vals[k]) && (ids[k + 1] < ids[k]))) {
        T tmp_val = vals[k];
        int tmp_id = ids[k];
        int parent_id2 = parent_ids[k];
        vals[k] = vals[k + 1];
        ids[k] = ids[k + 1];
        parent_ids[k] = parent_ids[k + 1];
        vals[k + 1] = tmp_val;
        ids[k + 1] = tmp_id;
        parent_ids[k + 1] = parent_id2;
      }
    }
  }
};

template <typename T, int K>
__device__ __forceinline__ TopK<T, K> reduce_topk_op(const TopK<T, K> &a,
                                                     const TopK<T, K> &b) {
  TopK<T, K> res = a;
  for (int i = 0; i < K; ++i) res.insert(b.vals[i], b.ids[i]);
  return res;
}

template <typename T, int K>
struct TopKSoftMax {
  DySoftMaxStruct softmax_md;
  TopK<T, K> topk;
};

template <typename T, int K>
__device__ __forceinline__ TopKSoftMax<T, K> reduce_topk_softmax_op(
    const TopKSoftMax<T, K> &a, const TopKSoftMax<T, K> &b) {
  TopKSoftMax<T, K> res;
  res.softmax_md = reduce_softmax_op(a.softmax_md, b.softmax_md);
  res.topk = reduce_topk_op(a.topk, b.topk);
  return res;
}

template <typename T, int K, int THREADBLOCK_SIZE>
__global__ void batch_topk(const int *topk_tmp_id_buf,
                           const T *topk_tmp_val_buf,
                           const int *step_ids,
                           const bool *stop_flags, // bs * beam_size
                           const int *seq_lens,
                           const int *end_ids,
                           int *id_buf,
                           T *val_buf,
                           int *parent_idx,
                           bool *stop_flags_out,
                           int *seq_lens_out,
                           int *step_ids_out) {
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x; // bs
  const int beam_size = K / 2;
  TopK<T, beam_size> partial;
  if (thread_id == 0) {
    for (int i = 0; i < beam_size; ++i) {
      partial.ids[i] = -1;
      partial.vals[i] = -FLT_MAX;
      partial.parent_ids[i] = -1;
    }

    int index = block_id * beam_size * K;
    if (step_ids[0] == 0) {
      for (int i = 0; i < K; i++) {
        partial.insert(
            (T)topk_tmp_val_buf[index + i], topk_tmp_id_buf[index + i], i / K);
      }
    } else {
      for (int i = 0; i < beam_size * K; i++) {
        partial.insert(
            (T)topk_tmp_val_buf[index + i], topk_tmp_id_buf[index + i], i / K);
      }
    }
    index = block_id * beam_size;
    for (int i = 0; i < beam_size; i++) {
      id_buf[index + i] = partial.ids[i];
      val_buf[index + i] = partial.vals[i];
      int parent_id = partial.parent_ids[i];
      parent_idx[index + i] = parent_id;
      stop_flags_out[index + i] = stop_flags[index + parent_id];
      seq_lens_out[index + i] = seq_lens[index + parent_id];
      step_ids_out[index + i] = step_ids[index + parent_id];
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
      printf("bi: %d, id: %d, val: %f, parent_id: %d\n", block_id,
             id_buf[index+i], val_buf[index+i], parent_id);
#endif
    }
  }
}

// early stop
template <typename T, int K, int THREADBLOCK_SIZE>
__global__ void batch_topk(const int *topk_tmp_id_buf,
                           const T *topk_tmp_val_buf,
                           const float *cum_log_probs,
                           const int *step_ids,
                           const bool *stop_flags, // bs * beam_size
                           const int *seq_lens,
                           const int *end_ids,
                           int *id_buf,
                           T *val_buf,
                           int *parent_idx,
                           bool *stop_flags_out,
                           int *seq_lens_out,
                           int *step_ids_out) {
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x; // bs
  const int beam_size = K / 2;
  TopK<T, beam_size> partial;
  if (thread_id == 0) {
    for (int i = 0; i < beam_size; ++i) {
      partial.ids[i] = -1;
      partial.vals[i] = -FLT_MAX;
      partial.parent_ids[i] = -1;
    }

    int index = block_id * beam_size * K;
    if (step_ids[0] == 0) {
      for (int i = 0; i < K; i++) {
        partial.insert(
            (T)topk_tmp_val_buf[index + i], topk_tmp_id_buf[index + i], i / K);
      }
    } else {
      for (int i = 0; i < beam_size * K; i++) {
        if (!stop_flags[block_id * beam_size + i / K]) {
          // if stop, this branch end, no longer update.
          partial.insert(
              (T)topk_tmp_val_buf[index + i], topk_tmp_id_buf[index + i], i / K);
        }
      }
    }
    index = block_id * beam_size;
    int stop_num = 0;
    for (int i = 0; i < beam_size; i++) {
      if (stop_flags[index + i]) {
        parent_idx[index + i] = i;
        id_buf[index + i] = end_ids[0];
        val_buf[index + i] = cum_log_probs[index + i];
        stop_flags_out[index + i] = stop_flags[index + i];
        seq_lens_out[index + i] = seq_lens[index + i];
        step_ids_out[index + i] = step_ids_out[index + i];
        stop_num++;
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
        printf("%d has end, bi: %d, stop_num: %d\n", index + i, block_id, stop_num);
#endif
      } else {
        int parent_id = partial.parent_ids[i - stop_num];
        parent_idx[index + i] = parent_id;
        id_buf[index + i] = partial.ids[i - stop_num];
        val_buf[index + i] = partial.vals[i - stop_num];
        stop_flags_out[index + i] = stop_flags[index + parent_id];
        seq_lens_out[index + i] = seq_lens[index + parent_id];
        step_ids_out[index + i] = step_ids[index + parent_id];
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
        printf("bi: %d, id: %d, val: %f, parent_id: %d\n", block_id,
              id_buf[index+i], val_buf[index+i], parent_id);
#endif
      }
    }
  }
}

template <typename T, int K, int THREADBLOCK_SIZE, int PACKED_TOP_KMD_SIZE>
__global__ void beam_search_softmax_topk_stage1(const T *logits,
                                                const bool *stop_flags,
                                                const int *end_ids,
                                                float *tmp_buffer,
                                                const int vocab_size,
                                                const bool fuse_softmax) {
  int thread_id = threadIdx.x;
  int vector_id = blockIdx.x;  // batch beam index.

  __shared__ float buf_s[PACKED_TOP_KMD_SIZE];

  const T MAX_T_VAL = FLT_MAX;

  const int v_local = (vocab_size + gridDim.y - 1) / gridDim.y;
  const int section_start = v_local * blockIdx.y;
  int section_end = section_start + v_local;
  section_end = (section_end > vocab_size) ? vocab_size : section_end;

  logits += vector_id * vocab_size;
  if (fuse_softmax) {
    typedef cub::BlockReduce<TopKSoftMax<T, K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopKSoftMax<T, K> partial;
    bool finish = stop_flags[vector_id];
    for (int i = 0; i < K; ++i) {
      partial.topk.ids[i] = -1;
      partial.topk.vals[i] = -MAX_T_VAL;
    }
    partial.softmax_md.logit = -MAX_T_VAL;
    partial.softmax_md.score = 0.0F;

    if (finish) {
#pragma unroll 1
      for (int elem_id = section_start + thread_id; elem_id < section_end;
          elem_id += THREADBLOCK_SIZE) {
        // if is_end, set to (MAX_T_VAL, 1)
        T elem = (elem_id == end_ids[0]) ? MAX_T_VAL : -MAX_T_VAL;
        DySoftMaxStruct new_elem{elem, 1.0F};
        partial.softmax_md = reduce_softmax_op(partial.softmax_md, new_elem);
        partial.topk.insert(elem, elem_id);
      }
    } else {
#pragma unroll 1
      for (int elem_id = section_start + thread_id; elem_id < section_end;
          elem_id += THREADBLOCK_SIZE) {
        T elem = logits[elem_id];
        DySoftMaxStruct new_elem{elem, 1.0F};
        partial.softmax_md = reduce_softmax_op(partial.softmax_md, new_elem);
        partial.topk.insert(elem, elem_id);
      }
    }

    TopKSoftMax<T, K> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_softmax_op<T, K>);

    if (thread_id == 0) {
      for (int i = 0; i < K; i++) {
        reinterpret_cast<int *>(buf_s)[i] = total.topk.ids[i];
        buf_s[K + i] = total.topk.vals[i];
      }
      buf_s[2 * K] = total.softmax_md.score;
      buf_s[2 * K + 1] = total.softmax_md.logit;
    }
  } else {
    typedef cub::BlockReduce<TopK<T, K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopK<T, K> partial;
    bool finish = stop_flags[vector_id];
    for (int i = 0; i < K; ++i) {
      partial.ids[i] = -1;
      partial.vals[i] = -MAX_T_VAL;
    }

    if (finish) {
#pragma unroll 1
      for (int elem_id = section_start + thread_id; elem_id < section_end;
          elem_id += THREADBLOCK_SIZE) {
        // if is_end, set to (end_id, 1)
        T elem = (elem_id == end_ids[0]) ? 0 : -MAX_T_VAL;
        partial.insert(elem, elem_id);
      }
    } else {
#pragma unroll 1
      for (int elem_id = section_start + thread_id; elem_id < section_end;
          elem_id += THREADBLOCK_SIZE) {
        T elem = logits[elem_id];
        partial.insert(elem, elem_id);
      }
    }

    TopK<T, K> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, K>);

    if (thread_id == 0) {
      for (int i = 0; i < K; i++) {
        reinterpret_cast<int *>(buf_s)[i] = total.ids[i];
        buf_s[K + i] = total.vals[i];
      }
    }
  }
  __syncthreads();
  for (int elem_id = thread_id; elem_id < PACKED_TOP_KMD_SIZE;
      elem_id += THREADBLOCK_SIZE) {
    tmp_buffer[blockIdx.x * PACKED_TOP_KMD_SIZE * gridDim.y +
              blockIdx.y * PACKED_TOP_KMD_SIZE + elem_id] = buf_s[elem_id];
  }
}

template <typename T, int K, int THREADBLOCK_SIZE>
__global__ void beam_search_softmax_topk_stage2(const float *tmp_buffer,
                                                const float *cum_log_probs,
                                                int *tmp_ids,
                                                T *tmp_vals,
                                                const int voc_parts,
                                                const int packed_top_kmd_size,
                                                const bool fuse_softmax,
                                                const float length_penalty,
                                                const int *step_ids) {
  const int vector_id = blockIdx.x;
  const int thread_id = threadIdx.x;
  const int PACKED_TOP_KMD_SIZE = packed_top_kmd_size;

  const T MAX_T_VAL = FLT_MAX;

  extern __shared__ char buf_s_[];
  float *buf_s = reinterpret_cast<float *>(buf_s_);
  tmp_buffer += vector_id * PACKED_TOP_KMD_SIZE * voc_parts;

   // Since cum_log_probs is the penalized values, need to be restored before accumulation.
  T previous_penalty = static_cast<T>(powf(step_ids[vector_id], length_penalty));
  T current_penalty = static_cast<T>(powf(step_ids[vector_id] + 1, length_penalty));

  if (fuse_softmax) {
    typedef cub::BlockReduce<TopKSoftMax<T, K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopKSoftMax<T, K> partial;
    for (int i = 0; i < K; ++i) {
      partial.topk.ids[i] = -1;
      partial.topk.vals[i] = -MAX_T_VAL;
    }
    partial.softmax_md.logit = -MAX_T_VAL;
    partial.softmax_md.score = 0.0F;

    for (int idx = thread_id; idx < PACKED_TOP_KMD_SIZE * voc_parts;
        idx += THREADBLOCK_SIZE) {
      buf_s[idx] = tmp_buffer[idx];
    }
    __syncthreads();

    if (threadIdx.x < voc_parts) {
      float *b_s = buf_s + thread_id * PACKED_TOP_KMD_SIZE;
      for (int i = 0; i < K; i++) {
        partial.topk.ids[i] = reinterpret_cast<int *>(b_s)[i];
        partial.topk.vals[i] = b_s[K + i];
      }
      partial.softmax_md.score = b_s[2 * K];
      partial.softmax_md.logit = b_s[2 * K + 1];
    }
    __syncthreads();

    TopKSoftMax<T, K> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_softmax_op<T, K>);

    if (thread_id == 0) {
      tmp_ids += vector_id * K;
      tmp_vals += vector_id * K;
      cum_log_probs += vector_id;

      float d_total_log = log(total.softmax_md.score);
      for (int i = 0; i < K; ++i) {
        // float val = expf((float)total.topk.vals[i] - total.softmax_md.logit -  d_total_log);
        float val = total.topk.vals[i] - total.softmax_md.logit - d_total_log;
        tmp_ids[i] = total.topk.ids[i];
        tmp_vals[i] = (val + cum_log_probs[0] * previous_penalty) / current_penalty;
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
        printf("vector_id: %d, vals: %f, logit: %f, d_total_log: %f, id: %d, val: %f, cum_log_probs: %f, res: %f\n", vector_id, total.topk.vals[i], total.softmax_md.logit, d_total_log, tmp_ids[i], val, cum_log_probs[0], tmp_vals[i]);
#endif
      }
    }
  } else {
    typedef cub::BlockReduce<TopK<T, K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopK<T, K> partial;
    for (int i = 0; i < K; ++i) {
      partial.ids[i] = -1;
      partial.vals[i] = -MAX_T_VAL;
    }

    for (int idx = thread_id; idx < PACKED_TOP_KMD_SIZE * voc_parts;
          idx += THREADBLOCK_SIZE) {
      buf_s[idx] = tmp_buffer[idx];
    }
    __syncthreads();

    if (threadIdx.x < voc_parts) {
      float *b_s = buf_s + thread_id * PACKED_TOP_KMD_SIZE;
      for (int i = 0; i < K; i++) {
        partial.ids[i] = reinterpret_cast<int *>(b_s)[i];
        partial.vals[i] = b_s[K + i];
      }
    }
    __syncthreads();

    TopK<T, K> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, K>);

    if (thread_id == 0) {
      tmp_ids += vector_id * K;
      tmp_vals += vector_id * K;
      cum_log_probs += vector_id;

      for (int i = 0; i < K; ++i) {
        float val = total.vals[i];
        tmp_ids[i] = total.ids[i];
        tmp_vals[i] = (val + cum_log_probs[0] * previous_penalty) / current_penalty;
      }
    }
  }
}

template <typename T, int K>
void invokeBeamSearchSoftmaxTopKStage2(const float *tmp_buffer,
                                       const float *cum_log_probs,
                                       int *ids,
                                       T *vals,
                                       const int batch_size,
                                       const int beam_size,
                                       const int voc_parts,
                                       const int packed_top_kmd_size,
                                       const bool fuse_softmax,
                                       const float length_penalty,
                                       const int *step_ids,
                                       cudaStream_t stream) {
  int smem_stage2_size = voc_parts * packed_top_kmd_size * sizeof(float);

  if (voc_parts <= 32) {
    beam_search_softmax_topk_stage2<T, K, 32>
        <<<batch_size * beam_size, 32, smem_stage2_size, stream>>>(
            tmp_buffer, cum_log_probs, ids, vals, voc_parts, packed_top_kmd_size, fuse_softmax, length_penalty, step_ids);
    return;
  }
  if (voc_parts <= 64) {
    beam_search_softmax_topk_stage2<T, K, 64>
        <<<batch_size * beam_size, 64, smem_stage2_size, stream>>>(
            tmp_buffer, cum_log_probs, ids, vals, voc_parts, packed_top_kmd_size, fuse_softmax, length_penalty, step_ids);
    return;
  }
  if (voc_parts <= 128) {
    beam_search_softmax_topk_stage2<T, K, 128>
        <<<batch_size * beam_size, 128, smem_stage2_size, stream>>>(
            tmp_buffer, cum_log_probs, ids, vals, voc_parts, packed_top_kmd_size, fuse_softmax, length_penalty, step_ids);
    return;
  }
}

__global__ void update_beam_offsets_kernel(
    const int *src_indir_cache,   // bs * bm * max_len
    const int *beam_ids,          // bs * bm
    const int *sequence_lengths,  // bs * bm
    const bool *stop_flags,
    const int *step_ids,
    int *tgt_indir_cache,
    const int batch_size,
    const int beam_size,
    const int max_seq_len,
    const int max_dec_len) {
  int time_step = threadIdx.x + blockIdx.x * blockDim.x;
  int bb_id = blockIdx.y;
  const int batch_id = bb_id / beam_size;
  const int beam_id = bb_id % beam_size;
  const int src_beam = beam_ids[bb_id];
  const int src_bb_id = batch_id * beam_size + src_beam;
  const int seq_len = sequence_lengths[src_bb_id];
  const int max_len = max_seq_len + max_dec_len;


  if (seq_len == 0 || time_step >= min(seq_len + 1, max_len)) {
    return;
  }
  // if (time_step >= max_len) {
  //   return;
  // }

  if (bb_id >= beam_size * batch_size) {
    return;
  }

  const uint tgt_offset =
      batch_id * beam_size * max_len + beam_id * max_len + time_step;
  const uint src_offset =
      batch_id * beam_size * max_len + src_beam * max_len + time_step;

  tgt_indir_cache[tgt_offset] = (time_step == sequence_lengths[src_bb_id])
                                    ? src_beam
                                    : src_indir_cache[src_offset];
}

void invokeUpdateBeamOffset(const int *src_indir_cache,
                            const int *beam_ids,
                            const int *sequence_lengths,
                            const bool *stop_flags,
                            const int *step_ids,
                            int *tgt_indir_cache,
                            const int batch_size,
                            const int beam_size,
                            const int max_seq_len,
                            const int max_dec_len,
                            cudaStream_t stream) {
  const dim3 block(32);
  const dim3 grid((max_seq_len + max_dec_len + block.x - 1) / block.x,
                  batch_size * beam_size);
  update_beam_offsets_kernel<<<grid, block, 0, stream>>>(src_indir_cache,
                                                         beam_ids,
                                                         sequence_lengths,
                                                         stop_flags,
                                                         step_ids,
                                                         tgt_indir_cache,
                                                         batch_size,
                                                         beam_size,
                                                         max_seq_len,
                                                         max_dec_len);
}

__global__ void update_cache_ids_kernel(
    const int *last_cache_ids,  // bs * bm * max_dec_len
    const int *beam_ids,        // bs * bm
    const int *ids_this_time,
    const int *sequence_lengths,  // bs * bm
    const bool *stop_flags,
    const int *step_ids,
    int *cache_ids,
    const int batch_size,
    const int beam_size,
    const int max_dec_len) {
  int time_step = threadIdx.x + blockIdx.x * blockDim.x;
  int bb_id = blockIdx.y;
  const int batch_id = bb_id / beam_size;
  const int beam_id = bb_id % beam_size;
  const int src_beam = beam_ids[bb_id];
  const int src_bb_id = batch_id * beam_size + src_beam;
  const int step = step_ids[src_bb_id];

  if (sequence_lengths[src_bb_id] == 0 || time_step >= min(step + 1, max_dec_len)) {
    return;
  }

  if (bb_id >= beam_size * batch_size) {
    return;
  }

  const uint tgt_offset =
      batch_id * beam_size * max_dec_len + beam_id * max_dec_len + time_step;
  const uint src_offset =
      batch_id * beam_size * max_dec_len + src_beam * max_dec_len + time_step;

  cache_ids[tgt_offset] =
      (time_step == step) ? ids_this_time[bb_id] : last_cache_ids[src_offset];
}

void invokeUpdateCacheIds(const int *last_cache_ids,
                          const int *beam_ids,
                          const int *sequence_lengths,
                          const int *ids_this_time,
                          const bool *stop_flags,
                          const int *step_ids,
                          int *cache_ids,
                          const int batch_size,
                          const int beam_size,
                          const int max_dec_len,
                          cudaStream_t stream) {
  const dim3 block(32);
  const dim3 grid((max_dec_len + block.x - 1) / block.x,
                  batch_size * beam_size);
  update_cache_ids_kernel<<<grid, block, 0, stream>>>(last_cache_ids,
                                                      beam_ids,
                                                      ids_this_time,
                                                      sequence_lengths,
                                                      stop_flags,
                                                      step_ids,
                                                      cache_ids,
                                                      batch_size,
                                                      beam_size,
                                                      max_dec_len);
}

template <typename T, int K, typename Context>
void invokeTopKSoftMaxLauncher(const Context &dev_ctx,
                               const T *log_probs,
                               const bool *stop_flags,
                               const int *sequence_lengths,
                               const float *cum_log_probs,
                               const int *step_ids,
                               const int *last_cache_ids,
                               const int *last_beam_offsets,
                               const int *end_ids,
                               float *out_cum_log_probs,
                               bool *stop_flags_out,
                               int *seq_lens_out,
                               int *step_ids_out,
                               int *ids,
                               int *tmp_ids,
                               T *tmp_vals,
                               int *parent_idx,
                               int *cache_ids,
                               int *beam_offsets,
                               const int batch_size,
                               const int beam_size,
                               const int vocab_size,
                               const int max_seq_len,
                               const int max_dec_len,
                               const bool fuse_softmax,
                               const bool early_stop,
                               const float length_penalty,
                               cudaStream_t stream) {
  // K = 2 * beam_size
  const int block_size = 128;
  int voc_parts = vocab_size / 1024;
  voc_parts = std::min(128, voc_parts);
  int packed_top_kmd_size = 2 * K;
  if (fuse_softmax) {
    packed_top_kmd_size += 2;
  }
  const int tmp_buffer_size =
      batch_size * beam_size * voc_parts * packed_top_kmd_size;
  DenseTensor tmp_buffer_tensor;
  tmp_buffer_tensor.Resize(phi::make_ddim({tmp_buffer_size}));
  dev_ctx.template Alloc<float>(&tmp_buffer_tensor);
  float *tmp_buffer = tmp_buffer_tensor.data<float>();

  dim3 grid(batch_size * beam_size, voc_parts);
  if (fuse_softmax) {
    cudaFuncSetAttribute(beam_search_softmax_topk_stage1<T, K, block_size, 2 * K + 2>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxL1);
    // （bs, bm, voc_parts, 2 * K + 2）
    beam_search_softmax_topk_stage1<T, K, block_size, 2 * K + 2>
        <<<grid, block_size, 0, stream>>>(
            log_probs, stop_flags, end_ids, tmp_buffer, vocab_size, fuse_softmax);
  } else {
    cudaFuncSetAttribute(beam_search_softmax_topk_stage1<T, K, block_size, 2 * K>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxL1);
    // （bs, bm, voc_parts, 2 * K）
    beam_search_softmax_topk_stage1<T, K, block_size, 2 * K>
        <<<grid, block_size, 0, stream>>>(
            log_probs, stop_flags, end_ids, tmp_buffer, vocab_size, fuse_softmax);
  }
  // (bs, bm, K)
  invokeBeamSearchSoftmaxTopKStage2<T, K>(tmp_buffer,
                                          cum_log_probs,
                                          tmp_ids,
                                          tmp_vals,
                                          batch_size,
                                          beam_size,
                                          voc_parts,
                                          packed_top_kmd_size,
                                          fuse_softmax,
                                          length_penalty,
                                          step_ids,
                                          stream);
  // (bs, bm)
  if (early_stop) {
    batch_topk<T, K, 32><<<batch_size, 32, 0, stream>>>(
      tmp_ids, 
      tmp_vals, 
      cum_log_probs,
      step_ids, 
      stop_flags,
      sequence_lengths,
      end_ids,
      ids, 
      out_cum_log_probs, 
      parent_idx,
      stop_flags_out,
      seq_lens_out,
      step_ids_out);
  } else {
    batch_topk<T, K, 32><<<batch_size, 32, 0, stream>>>(
      tmp_ids, 
      tmp_vals, 
      step_ids, 
      stop_flags,
      sequence_lengths,
      end_ids,
      ids, 
      out_cum_log_probs, 
      parent_idx,
      stop_flags_out,
      seq_lens_out,
      step_ids_out);
  }
  invokeUpdateBeamOffset(last_beam_offsets,
                         parent_idx,
                         sequence_lengths,
                         stop_flags,
                         step_ids,
                         beam_offsets,
                         batch_size,
                         beam_size,
                         max_seq_len,
                         max_dec_len,
                         stream);
  invokeUpdateCacheIds(last_cache_ids,
                       parent_idx,
                       sequence_lengths,
                       ids,
                       stop_flags,
                       step_ids,
                       cache_ids,
                       batch_size,
                       beam_size,
                       max_dec_len,
                       stream);
}

template <typename T, typename Context>
void invokeTopkSoftMax(const Context &dev_ctx,
                       const T *log_probs,
                       const bool *stop_flags,
                       const int *sequence_lengths,
                       const float *cum_log_probs,
                       const int *step_ids,
                       const int *last_cache_ids,
                       const int *last_beam_offsets,
                       const int *end_ids,
                       float *out_cum_log_probs,
                       bool *stop_flags_out,
                       int *seq_lens_out,
                       int *step_ids_out,
                       int *ids,
                       int *tmp_ids,
                       T *tmp_vals,
                       int *parent_idx,
                       int *cache_ids,
                       int *beam_offsets,
                       const int batch_size,
                       const int beam_size,
                       const int vocab_size,
                       const int max_seq_len,
                       const int max_dec_len,
                       const bool fuse_softmax,
                       const bool early_stop,
                       const float length_penalty,
                       cudaStream_t stream) {
  switch (beam_size) {
    CASE_K(1);
    CASE_K(2);
    CASE_K(3);
    CASE_K(4);
    CASE_K(5);
    CASE_K(6);
    CASE_K(7);
    CASE_K(8);
    CASE_K(9);
    CASE_K(10);
    CASE_K(11);
    CASE_K(12);
    CASE_K(13);
    CASE_K(14);
    CASE_K(15);
    CASE_K(16);
    CASE_K(20);
    CASE_K(30);
    CASE_K(50);
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "beam_size = %d is unsupport!", beam_size));
  }
}

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
                             float length_penalty,
                             DenseTensor *ids_this_time,
                             DenseTensor *out_cum_scores,
                             DenseTensor *cache_ids,
                             DenseTensor *beam_offsets,
                             DenseTensor *parent_idx,
                             DenseTensor *stop_flags_out,
                             DenseTensor *seq_lens_out,
                             DenseTensor *step_ids_out) {
  const auto &logits_dims = logits.dims();
  int bs = logits_dims[0];
  int batch_size = bs / beam_size;
  int vocab_size = logits_dims[1];

  dev_ctx.template Alloc<int>(ids_this_time);
  dev_ctx.template Alloc<int>(cache_ids);
  dev_ctx.template Alloc<int>(beam_offsets);
  dev_ctx.template Alloc<int>(parent_idx);
  dev_ctx.template Alloc<T>(out_cum_scores);
  dev_ctx.template Alloc<bool>(stop_flags_out);
  dev_ctx.template Alloc<int>(seq_lens_out);
  dev_ctx.template Alloc<int>(step_ids_out);

  phi::Copy(dev_ctx, last_cache_ids, dev_ctx.GetPlace(), false, cache_ids);
  phi::Copy(
      dev_ctx, last_beam_offsets, dev_ctx.GetPlace(), false, beam_offsets);
  phi::Copy(
      dev_ctx, stop_flags, dev_ctx.GetPlace(), false, stop_flags_out);
  phi::Copy(
      dev_ctx, sequence_lengths, dev_ctx.GetPlace(), false, seq_lens_out);
  phi::Copy(
      dev_ctx, step_ids, dev_ctx.GetPlace(), false, step_ids_out);

  const int tmp_size = batch_size * beam_size * beam_size * 2;
  DenseTensor tmp_topk_id, tmp_topk_val;
  tmp_topk_id.Resize(phi::make_ddim({tmp_size}));
  dev_ctx.template Alloc<int>(&tmp_topk_id);
  tmp_topk_val.Resize(phi::make_ddim({tmp_size}));
  dev_ctx.template Alloc<T>(&tmp_topk_val);

  invokeTopkSoftMax(dev_ctx,
                    logits.data<T>(),
                    stop_flags.data<bool>(),
                    sequence_lengths.data<int>(),
                    cum_scores.data<T>(),
                    step_ids.data<int>(),
                    last_cache_ids.data<int>(),
                    last_beam_offsets.data<int>(),
                    end_ids.data<int>(),
                    out_cum_scores->data<T>(),
                    stop_flags_out->data<bool>(),
                    seq_lens_out->data<int>(),
                    step_ids_out->data<int>(),
                    ids_this_time->data<int>(),
                    tmp_topk_id.data<int>(),
                    tmp_topk_val.data<T>(),
                    parent_idx->data<int>(),
                    cache_ids->data<int>(),
                    beam_offsets->data<int>(),
                    batch_size,
                    beam_size,
                    vocab_size,
                    max_seq_len,
                    max_dec_len,
                    fuse_softmax,
                    early_stop,
                    length_penalty,
                    dev_ctx.stream());
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(beam_search_softmax,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::BeamSearchSoftmaxKernel,
                   float) {}  // only supports float
