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
// #define DEBUG_PRINT_LINEAR_SHAPE
// #define DEBUG_TMPROFILE_WEIGHT_ONLY
#include "paddle/fluid/operators/fused/fused_multi_transformer_op.h"
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
#include "paddle/fluid/platform/timer.h"
#endif
#include "paddle/fluid/operators/fused/attn_gemm.h"
#include "paddle/fluid/operators/fused/moe_expert_gemm.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"
#include "paddle/phi/kernels/gpu/fused_moe_kernel.cu.h"
#include "paddle/phi/kernels/weight_only_linear_kernel.h"

PADDLE_DEFINE_EXPORTED_bool(enable_moe_gemm_cutlass,
                            false,
                            "enable moe gemm cutlass ,default false");
namespace paddle {
namespace operators {
using Tensor = phi::DenseTensor;
// #define _DEBUG_FUSED_MULTI_TRANSFORMER
inline bool CheckFlashAttn(const phi::GPUContext &dev_ctx,
                           const phi::DenseTensor &x) {
  int dev = dev_ctx.GetPlace().GetDeviceId();
  if (!paddle::platform::IsSupportFlashAttn(dev)) {
    return false;
  }
  return (x.dtype() == DataType::FLOAT16);
}
template <typename T>
class FusedMultiTransformerMoeWeightOnlyOpKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    auto &dev_ctx = ctx.cuda_device_context();
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
    platform::Timer all_tm, other_tm, trans_tm;
    platform::Timer qkv_tm, fmha_tm, out_linear_tm;
    platform::Timer expert_tm, ln_tm, gate_tm;
    platform::Timer gate_nccl_tm, gather_tm, scatter_tm;
    all_tm.Start();
    other_tm.Start();
#endif
    auto *time_step = ctx.Input<Tensor>("TimeStep");
    // 0. input  [batch_size, seq_len, dim_embed]
    auto *input_x = ctx.Input<Tensor>("X");
    const auto input_x_dims = input_x->dims();
    int bsz = input_x_dims[0];
    int seq_len = input_x_dims[1];
    int dim_embed = input_x_dims[2];
    int bsz_seq = bsz * seq_len;
    if (bsz_seq == 0) {
      return;
    }
    // LOG(INFO) << "intput X: bsz: " << bsz << ", seq_len: " << seq_len << ",
    // dim_embed: " << dim_embed;
    const std::string act_method = ctx.Attr<std::string>("act_method");
    auto *sequence_lengths = ctx.Input<Tensor>("SeqLengths");  // nullptr
    auto *beam_cache_offset = ctx.Input<Tensor>("BeamCacheOffset");
    int beam_size = 1;
    if (beam_cache_offset) {
      beam_size = beam_cache_offset->dims()[1];
    }
    // LOG(INFO) << "beam_size: " << beam_size;

    auto *out = ctx.Output<Tensor>("Out");
    dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));

    // 1. layer norm
    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");
    if (!pre_layer_norm) {
      VLOG(0) << "not support post layer norm!";
      return;
    }
    const float epsilon = ctx.Attr<float>("epsilon");
    auto ln_scales = ctx.MultiInput<Tensor>("LnScale");
    auto ln_biases = ctx.MultiInput<Tensor>("LnBias");

    auto ln_compute = AttnLayerNorm<T>(dev_ctx, epsilon, bsz_seq, dim_embed);
    Tensor ln_mean, ln_var;
    ln_mean.Resize({{bsz_seq}});
    auto *ln_mean_data =
        dev_ctx.Alloc<U>(&ln_mean, ln_mean.numel() * sizeof(U));
    ln_var.Resize({{bsz_seq}});
    auto *ln_var_data = dev_ctx.Alloc<U>(&ln_var, ln_var.numel() * sizeof(U));

    // 2. qkv
    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto qkv_weights = ctx.MultiInput<Tensor>("QKVW");
    auto qkv_scales = ctx.MultiInput<Tensor>("QKVWScale");
    auto qkv_biases = ctx.MultiInput<Tensor>("QKVBias");
    const std::string weight_dtype = ctx.Attr<std::string>("weight_dtype");
    const auto qkv_w_dims = qkv_weights[0]->dims();
    int num_head = qkv_w_dims[1];
    int dim_head = qkv_w_dims[2];
    const bool is_int4 = (weight_dtype == "int4");
    if (is_int4) {
      // int4 weight: [3, num_head, dim_head / 2, dim_embed]
      dim_head = dim_head * 2;
    }
    int hidden_size = num_head * dim_head;
    int qkv_output_size = 3 * hidden_size;
    // weight only gemm
    auto weight_only_gemm = AttnMatMulWeightOnly<T>(dev_ctx, is_int4);
    int default_act = weight_only_gemm.GetActivation("none");
    int expert_act = weight_only_gemm.GetActivation(act_method);

#ifndef PADDLE_WITH_CUTLASS
    PADDLE_ENFORCE_EQ(FLAGS_enable_moe_gemm_cutlass, false,
                      "not support cutlass fused moe gemm please disable "
                      "FLAGS_enable_moe_gemm_cutlass");
#endif
    auto moe_expert_gemm = MoeExpertGemmWeightOnly<T>(dev_ctx, is_int4);

    Tensor qkv_out;
    qkv_out.Resize({{bsz, seq_len, 3, num_head, dim_head}});
    auto *qkv_out_data =
        dev_ctx.Alloc<T>(&qkv_out, qkv_out.numel() * sizeof(T));

    // 3. fmha
    auto dropout_implementation =
        ctx.Attr<std::string>("dropout_implementation");
    AttnDropoutParam attn_param(
        true, dropout_implementation, 0.0, true, true, 0, nullptr);
    auto fmha_compute =
        FMHARef<T>(dev_ctx, bsz, seq_len, num_head, dim_head, attn_param);

    // check support flash attn
    bool is_support_flash_attn = CheckFlashAttn(dev_ctx, *input_x);
    auto fmha_fa_compute = FlashAttnFMHARef<plat::float16>(
        dev_ctx, bsz, seq_len, num_head, dim_head, attn_param);
    auto *src_mask = ctx.Input<Tensor>("SrcMask");
    auto cache_kvs = ctx.MultiInput<Tensor>("CacheKV");
    auto cache_kv_outs = ctx.MultiOutput<Tensor>("CacheKVOut");

    int time_step_cpu = 0;
    if (time_step) {
      time_step_cpu = src_mask->dims()[3] - 1;
    }

    auto out_seq_len = seq_len;
    if (time_step) {
      PADDLE_ENFORCE_GT(
          time_step_cpu,
          0,
          platform::errors::PreconditionNotMet(
              "The value of time_step must > 0, but now is %d", time_step_cpu));
      PADDLE_ENFORCE_EQ(
          seq_len,
          1,
          platform::errors::PreconditionNotMet(
              "In decode stage, the seq_len of input must be 1, but now is %d",
              seq_len));
      out_seq_len += time_step_cpu;
    }

    Tensor transpose_out_2, qk_out;
    transpose_out_2.Resize({{3, bsz, num_head, seq_len, dim_head}});
    auto *transpose_out_2_data =
        dev_ctx.Alloc<T>(&transpose_out_2, transpose_out_2.numel() * sizeof(T));

    Tensor softmax_out;
    Tensor attn_dropout_mask_out, attn_dropout_out;
    Tensor qktv_out, fmha_out;
    if (!is_support_flash_attn) {
      qk_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
      auto *qk_out_data = dev_ctx.Alloc<T>(&qk_out, qk_out.numel() * sizeof(T));

      softmax_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
      auto *softmax_out_data =
          dev_ctx.Alloc<T>(&softmax_out, softmax_out.numel() * sizeof(T));
      qktv_out.Resize({{bsz, num_head, seq_len, dim_head}});
      auto *qktv_out_data =
          dev_ctx.Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
    }

    fmha_out.Resize({{bsz, seq_len, num_head, dim_head}});
    auto *fmha_out_data =
        dev_ctx.Alloc<T>(&fmha_out, fmha_out.numel() * sizeof(T));

    // 4. out_linear
    auto out_linear_weights = ctx.MultiInput<Tensor>("OutLinearW");
    auto out_linear_scales = ctx.MultiInput<Tensor>("OutLinearWScale");
    auto out_linear_biases = ctx.MultiInput<Tensor>("OutLinearBias");
    int ring_id = ctx.Attr<int>("ring_id");

    // 5. ln(residual + bias), pre layernorm in ffn/moe
    DropoutParam dropout_param(false, 0, true, true, 0.0, nullptr, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        dev_ctx, bsz_seq, dim_embed, dropout_param, epsilon);
    auto ffn_ln_scales = ctx.MultiInput<Tensor>("FFNLnScale");
    auto ffn_ln_biases = ctx.MultiInput<Tensor>("FFNLnBias");
    Tensor bias_dropout_residual_out, dropout_mask_out;
    T *bias_dropout_residual_out_data = nullptr;
    bias_dropout_residual_out.Resize({{bsz_seq, dim_embed}});
    bias_dropout_residual_out_data =
        dev_ctx.Alloc<T>(&bias_dropout_residual_out,
                         bias_dropout_residual_out.numel() * sizeof(T));
    uint8_t *dropout_mask_out_data = nullptr;

    // 6. moe layer: gate / expert_w & b / some attrs
    auto gate_weights = ctx.MultiInput<Tensor>("GateWeight");
    auto gate_biases = ctx.MultiInput<Tensor>("GateBias");
    auto expert_weights1 = ctx.MultiInput<Tensor>("ExpertWeight1");
    auto expert_scales1 = ctx.MultiInput<Tensor>("ExpertWeight1Scale");
    auto expert_biases1 = ctx.MultiInput<Tensor>("ExpertBias1");
    auto expert_weights2 = ctx.MultiInput<Tensor>("ExpertWeight2");
    auto expert_scales2 = ctx.MultiInput<Tensor>("ExpertWeight2Scale");
    auto expert_biases2 = ctx.MultiInput<Tensor>("ExpertBias2");
    // expert_weights1: int8 [dim_feedforward, dim_embed]  int8 [dim_feedforward
    // / 2, dim_embed]
    int dim_feedforward = expert_weights1[0]->dims()[0];
    if (is_int4) {
      dim_feedforward = dim_feedforward * 2;
    }

    int topk = ctx.Attr<int>("topk");
    int mp_size = ctx.Attr<int>("mp_size");
    int mp_rank = ctx.Attr<int>("mp_rank");
    int num_expert = ctx.Attr<int>("num_expert");
    int world_size = ctx.Attr<int>("world_size");
    int moe_ring_id = ctx.Attr<int>("moe_ring_id");
    bool approximate = ctx.Attr<bool>("approximate");

    int tot_expert = world_size * num_expert;
    // after slice, bsz_seq should be change
    int sliced_bsz_seq = bsz_seq;
    int start = 0;
    int end = 0;
    if (mp_size > 1) {
      start = bsz_seq / world_size * mp_rank;
      end = std::min(start + bsz_seq / world_size, bsz_seq);
      sliced_bsz_seq = end - start;
    }
    int out_batch_size = sliced_bsz_seq * topk;
    // slice
    Tensor sliced_inp;
    sliced_inp.Resize({{sliced_bsz_seq, dim_embed}});
    dev_ctx.Alloc<T>(&sliced_inp, sliced_inp.numel() * sizeof(T));
    // gate linear
    Tensor gate_out;
    gate_out.Resize({{sliced_bsz_seq, tot_expert}});
    dev_ctx.Alloc<T>(&gate_out, gate_out.numel() * sizeof(T));
    // topk
    Tensor topk_value, topk_idx;
    topk_value.Resize({{sliced_bsz_seq, topk}});
    dev_ctx.Alloc<T>(&topk_value, topk_value.numel() * sizeof(T));
    topk_idx.Resize({{sliced_bsz_seq, topk}});
    dev_ctx.Alloc<int64_t>(&topk_idx, topk_idx.numel() * sizeof(int64_t));
    // local expert count, global expert count
    Tensor local_expert_count, global_expert_count;
    local_expert_count.Resize({{tot_expert}});
    global_expert_count.Resize({{tot_expert}});
    dev_ctx.Alloc<int64_t>(&local_expert_count,
                           local_expert_count.numel() * sizeof(int64_t));
    dev_ctx.Alloc<int64_t>(&global_expert_count,
                           global_expert_count.numel() * sizeof(int64_t));
    // fwd_expert_count, fwd_batch_size
    Tensor fwd_expert_count, fwd_expert_csum_len;
    Tensor fwd_expert_csum_len_cpu;
    fwd_expert_count.Resize({{num_expert}});
    fwd_expert_csum_len.Resize({{num_expert + 1}});
    dev_ctx.Alloc<int64_t>(&fwd_expert_count,
                           fwd_expert_count.numel() * sizeof(int64_t));
    dev_ctx.Alloc<int64_t>(&fwd_expert_csum_len,
                           fwd_expert_csum_len.numel() * sizeof(int64_t));
    phi::funcs::set_constant<int64_t>(
        dev_ctx, &fwd_expert_csum_len, static_cast<int64_t>(0));

    // pos, temp pos
    Tensor pos, ins_pos;
    pos.Resize({{out_batch_size}});
    ins_pos.Resize({{out_batch_size}});
    dev_ctx.Alloc<int64_t>(&pos, pos.numel() * sizeof(int64_t));
    if (topk > 1) {
      dev_ctx.Alloc<int64_t>(&ins_pos, ins_pos.numel() * sizeof(int64_t));
    }
    // cumsum
    Tensor lec_cum;
    lec_cum.Resize({{tot_expert}});
    dev_ctx.Alloc<int64_t>(&lec_cum, lec_cum.numel() * sizeof(int64_t));
    // fused moe ffn tmp out
    Tensor index_select_out;
    index_select_out.Resize({{out_batch_size, dim_embed}});
    dev_ctx.Alloc<T>(&index_select_out, index_select_out.numel() * sizeof(T));
    Tensor global_gather_out;
    global_gather_out.Resize({{out_batch_size, dim_embed}});
    dev_ctx.Alloc<T>(&global_gather_out, global_gather_out.numel() * sizeof(T));
    Tensor moe_gather_out;
    moe_gather_out.Resize({{out_batch_size, dim_embed}});
    dev_ctx.Alloc<T>(&moe_gather_out, moe_gather_out.numel() * sizeof(T));
    Tensor bmm_out;
    bmm_out.Resize({{sliced_bsz_seq, 1, dim_embed}});
    dev_ctx.Alloc<T>(&bmm_out, bmm_out.numel() * sizeof(T));
    Tensor all_gather_out;
    all_gather_out.Resize({{bsz_seq, dim_embed}});
    dev_ctx.Alloc<T>(&all_gather_out, all_gather_out.numel() * sizeof(T));
    // topk tensor
    Tensor topk_tensor;
    topk_tensor.Resize({{1}});
    dev_ctx.Alloc<int64_t>(&topk_tensor, topk_tensor.numel() * sizeof(int64_t));
    phi::FullKernel<int64_t, phi::GPUContext>(
        dev_ctx, {1}, topk, pos.dtype(), &topk_tensor);
    // moe nccl
    phi::NCCLMoECollective moe_pg(dev_ctx, moe_ring_id, num_expert);

    Tensor buf0, moe_out;
    buf0.Resize({{bsz_seq, dim_embed}});
    dev_ctx.Alloc<T>(&buf0, buf0.numel() * sizeof(T));
    moe_out.ShareDataWith(*out);
    moe_out.Resize({{bsz_seq, dim_embed}});
    // expert
    Tensor expert_out1;
    Tensor global_scatter_out;
    Tensor all_expert_out;

    const T *x_data = input_x->data<T>();
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
    dev_ctx.Wait();
    other_tm.Pause();
#endif
    int layers = qkv_weights.size();
    for (int i = 0; i < layers; ++i) {
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      trans_tm.Resume();
#endif
      // step1. layer_norm, only layer 0
      if (i == 0) {
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
        ln_tm.Resume();
#endif
        auto *ln_scale_data = ln_scales[i]->data<U>();
        auto *ln_bias_data = ln_biases[i]->data<U>();
        // TODO(wangxi): can remove mean var in inference
        ln_compute.ComputeForward(x_data,
                                  ln_scale_data,
                                  ln_bias_data,
                                  buf0.data<T>(),
                                  ln_mean_data,
                                  ln_var_data);
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
        dev_ctx.Wait();
        ln_tm.Pause();
#endif
      }

      // step2. qkv
      const Tensor *qkv_bias = qkv_biases.size() > 0 ? qkv_biases[i] : nullptr;
      // NOTE: in decoder stage, bias is fused in fmha
      // M = batch_size * seq_len, N = 3 * num_head * dim_head, K = dim_embed
#ifdef DEBUG_PRINT_LINEAR_SHAPE
      VLOG(0) << "layer id=" << i << ", qkv input=" << buf0.dims()
              << ", weight=" << qkv_weights[i]->dims()
              << ", output=" << qkv_out.dims();
#endif
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      dev_ctx.Wait();
      qkv_tm.Resume();
#endif
      weight_only_gemm.Linear(buf0,
                              *qkv_weights[i],
                              (time_step == nullptr) ? qkv_bias : nullptr,
                              *qkv_scales[i],
                              bsz_seq,          // M
                              qkv_output_size,  // N
                              dim_embed,        // K
                              default_act,      // none
                              &qkv_out);
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      dev_ctx.Wait();
      qkv_tm.Pause();
      fmha_tm.Resume();
#endif
      // step3. fmha
      const Tensor *cache_kv = cache_kvs.size() > 0 ? cache_kvs[i] : nullptr;
      Tensor *cache_kv_out = cache_kv ? cache_kv_outs[i] : nullptr;

      if (time_step) {  // generation decoder stage
        // [2, batch_size, num_head, max_seq_len, head_size]
        int max_seq_len = cache_kv->dims()[3];
        fmha<T>(dev_ctx,
                qkv_out,
                *qkv_bias,
                *src_mask,
                sequence_lengths,
                nullptr,
                beam_cache_offset,
                cache_kv_out,
                &fmha_out,
                bsz,
                beam_size,
                max_seq_len,
                num_head,
                dim_head,
                time_step_cpu,
                0,
                1. / sqrt(dim_head));
      } else if (cache_kv_out) {  // generation encoder stage
        if (is_support_flash_attn) {
          fmha_fa_compute.ComputeForward(qkv_out,
                                         nullptr,
                                         src_mask,
                                         &transpose_out_2,
                                         nullptr,
                                         &softmax_out,  // softmax_lse_out
                                         &attn_dropout_mask_out,  // seek_offset
                                         &attn_dropout_out,       // softmax_out
                                         &fmha_out);
          // input: [bs, seq_len, 3, num_head, head_dim]
          // output: [3, bs, num_head, seq_len, head_dim]
          std::vector<int> perm_1 = {2, 0, 3, 1, 4};
          transpose_out_2.Resize({{3, bsz, num_head, seq_len, dim_head}});
          TransposeGPUKernelDriver<T>(
              dev_ctx, qkv_out, perm_1, &transpose_out_2);
        } else {
          fmha_compute.ComputeForward(qkv_out,
                                      nullptr,
                                      src_mask,
                                      &transpose_out_2,
                                      nullptr,
                                      &qk_out,
                                      nullptr,
                                      &softmax_out,
                                      &attn_dropout_mask_out,
                                      &attn_dropout_out,
                                      &qktv_out,
                                      &fmha_out);
        }
        // [3, bsz, num_head, seq_len, head_dim]
        T *qkv_data = transpose_out_2_data;
        int64_t q_size = bsz * seq_len * num_head * dim_head;
        int64_t k_size = q_size;
        const T *q_ptr = qkv_data;
        const T *k_ptr = q_ptr + q_size;
        const T *v_ptr = k_ptr + k_size;

        // [2, bsz, num_head, max_seq_len, head_dim]
        int max_seq_len = cache_kv_out->dims()[3];
        T *cache_kv_data = cache_kv_out->data<T>();
        int64_t cache_k_size = bsz * num_head * max_seq_len * dim_head;

        T *cache_k_ptr = cache_kv_data;
        T *cache_v_ptr = cache_kv_data + cache_k_size;

        write_cache_kv<T>(dev_ctx,
                          cache_k_ptr,
                          cache_v_ptr,
                          k_ptr,
                          v_ptr,
                          bsz,
                          num_head,
                          seq_len,
                          max_seq_len,
                          dim_head);
      } else {  // not generation
        VLOG(0) << "not support!";
      }
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      dev_ctx.Wait();
      fmha_tm.Pause();
      out_linear_tm.Resume();
#endif
      // 输出到buf0
      // M = batch_size * seq_len, N = dim_embed, K = num_head * dim_head
#ifdef DEBUG_PRINT_LINEAR_SHAPE
      VLOG(0) << "layer id=" << i << ", out linear input=" << fmha_out.dims()
              << ", weight=" << out_linear_weights[i]->dims()
              << ", output=" << buf0.dims();
#endif
      weight_only_gemm.Linear(fmha_out,
                              *out_linear_weights[i],
                              nullptr,
                              *out_linear_scales[i],
                              bsz_seq,      // M
                              dim_embed,    // N
                              hidden_size,  // K
                              default_act,  // none
                              &buf0);
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      dev_ctx.Wait();
      out_linear_tm.Pause();
#endif
      // mp need allreduce
      if (mp_size > 1) {
        phi::AllReduce<T>(buf0, ring_id, buf0.numel(), dev_ctx);
      }

      // step5. ln(residual + dropout(input + bias))，在MHA里的
      auto *ln_scale_data = ffn_ln_scales[i]->data<U>();
      auto *ln_bias_data = ffn_ln_biases[i]->data<U>();
      auto *out_linear_bias_data = out_linear_biases[i]->data<T>();
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      ln_tm.Resume();
#endif
      // pre layer norm : bias_dropout_residual_out is residual
      fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
          dev_ctx,
          buf0.data<T>(),
          x_data,  // residual, moe out
          out_linear_bias_data,
          ln_scale_data,
          ln_bias_data,
          bias_dropout_residual_out_data,
          dropout_mask_out_data,
          buf0.data<T>(),  // output to buf0
          ln_mean_data,
          ln_var_data);
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      dev_ctx.Wait();
      ln_tm.Pause();
#endif
      // moe
      // step2 resize and slice ln_out
      if (mp_size > 1) {
        sliced_inp = buf0.Slice(start, end);
      } else {
        sliced_inp = buf0;
      }
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      gate_tm.Resume();
#endif
      // step3 gate & topk
      phi::MatMulAndAdd<T>(dev_ctx,
                           gate_weights[i],
                           &sliced_inp,
                           gate_biases[i],
                           false,
                           false,
                           true,  //  compute bias
                           &gate_out,
                           &gate_out);
      phi::TopkKernel<T, phi::GPUContext>(dev_ctx,
                                          gate_out,
                                          topk,  // scalar
                                          -1,
                                          true,
                                          false,
                                          &topk_value,
                                          &topk_idx);

      // phi::CheckTopkIndex<int64_t>(dev_ctx, topk_idx);
      // step4 prepare forward
      // step4.1 number count
      phi::NumberCountKernel<int64_t, phi::GPUContext>(
          dev_ctx, topk_idx, tot_expert, &local_expert_count);

      // step4.2 all_to_all
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      dev_ctx.Wait();
      gate_nccl_tm.Resume();
#endif
      if (world_size > 1) {
#ifdef DEBUG_PRINT_LINEAR_SHAPE
        VLOG(0) << "layer id=" << i << ", begin all2all";
#endif
        moe_pg.AllToAll<int64_t>(local_expert_count, global_expert_count);
      } else {
        global_expert_count = local_expert_count;
      }
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      dev_ctx.Wait();
      gate_nccl_tm.Pause();
#endif
      // global expert count resize
      global_expert_count.Resize({{world_size, num_expert}});
      // fwd expert count
      phi::SumKernel<int64_t, phi::GPUContext>(dev_ctx,
                                               global_expert_count,
                                               phi::IntArray({0}),
                                               global_expert_count.dtype(),
                                               false,
                                               &fwd_expert_count);
      // fwd batch size
      phi::CumsumTensorValue<int64_t>(
          dev_ctx, fwd_expert_count, &fwd_expert_csum_len, 1);
      // step4.3 cumsum & assign pos
      phi::CumsumTensorValue<int64_t>(dev_ctx, local_expert_count, &lec_cum);
      // 1. assign pos and input ins pos
      phi::AssignInsAndPosCompute<int64_t>(
          dev_ctx, &lec_cum, &topk_idx, &pos, out_batch_size, topk, &ins_pos);

      framework::TensorCopy(
          fwd_expert_csum_len, platform::CPUPlace(), &fwd_expert_csum_len_cpu);
      dev_ctx.Wait();
      int fwd_bsz = fwd_expert_csum_len_cpu.data<int64_t>()[num_expert];

      global_scatter_out.Resize({{fwd_bsz, dim_embed}});
      dev_ctx.Alloc<T>(&global_scatter_out,
                       global_scatter_out.numel() * sizeof(T));

      all_expert_out.Resize({{fwd_bsz, dim_embed}});
      dev_ctx.Alloc<T>(&all_expert_out, all_expert_out.numel() * sizeof(T));

      // step 5, MOEScatter
      // step 5.1, index select
      phi::IndexSelectKernel<T, phi::GPUContext>(
          dev_ctx, sliced_inp, ins_pos, 0, &index_select_out);
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      dev_ctx.Wait();
      gate_tm.Pause();

      dev_ctx.Wait();
      scatter_tm.Resume();
#endif
      if (world_size > 1) {
#ifdef DEBUG_PRINT_LINEAR_SHAPE
        VLOG(0) << "layer id=" << i << ", begin scatter x=" << index_select_out.dims();
#endif
        moe_pg.Scatter<T>(&index_select_out,
                          local_expert_count,
                          global_expert_count,
                          &global_scatter_out);
      } else {
        global_scatter_out = index_select_out;
      }
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      dev_ctx.Wait();
      scatter_tm.Pause();

      dev_ctx.Wait();
      expert_tm.Resume();
#endif
#ifdef DEBUG_PRINT_LINEAR_SHAPE
      VLOG(0) << "layer id=" << i << ", begin expert fwd_bsz=" << fwd_bsz
              << ", dim_feedforward=" << dim_feedforward
              << ", dim_embed=" << dim_embed;
#endif
      // step 6, Expert Computation
      if (fwd_bsz != 0) {
        // encoder, use matmul
        if (FLAGS_enable_moe_gemm_cutlass) {
          int expert_idx = i * num_expert;
#ifdef DEBUG_PRINT_LINEAR_SHAPE
          std::ostringstream ostr;
          int64_t *pnum = fwd_expert_csum_len_cpu.data<int64_t>();
          for (int j = 0; j <= num_expert; ++j) {
            ostr << pnum[j] << ",";
          }
          VLOG(0)
              << "layer id=" << i << ", expert_idx=" << expert_idx
              << ", numel=" << fwd_expert_count.numel()
              << ", dim_feedforward=" << dim_feedforward
              << ", dim_embed=" << dim_embed << ", num_expert=" << num_expert
              << ", global_scatter_out=" << global_scatter_out.dims()
              << ", expert_weights1=" << expert_weights1[expert_idx]->dims()
              << ", start ptr="
              << (int64_t)(expert_weights1[expert_idx]->data()) << ", end ptr="
              << (int64_t)(expert_weights1[expert_idx + num_expert - 1]->data())
              << ", numel=" << expert_weights1[expert_idx]->numel()
              << ", expert_weights2=" << expert_weights2[expert_idx]->dims()
              << ", expert nums=" << ostr.str();
#endif
          // step 6.1, expert gemm
          expert_out1.Resize({{fwd_bsz, dim_feedforward}});
          dev_ctx.Alloc<T>(&expert_out1, expert_out1.numel() * sizeof(T));

          moe_expert_gemm.moe_gemm(fwd_expert_csum_len,
                                   global_scatter_out,
                                   expert_weights1[expert_idx],
                                   expert_scales1[expert_idx],
                                   expert_biases1[expert_idx],
                                   expert_weights2[expert_idx],
                                   expert_scales2[expert_idx],
                                   expert_biases2[expert_idx],
                                   fwd_bsz,
                                   dim_feedforward,
                                   dim_embed,
                                   num_expert,
                                   expert_act,
                                   default_act,
                                   &expert_out1,
                                   &all_expert_out);
        } else {
          int last_index = 0;
          int64_t *csum_len = fwd_expert_csum_len_cpu.data<int64_t>();
          for (int idx = 0; idx < num_expert; idx++) {
            int end = csum_len[idx + 1];
            int cur_expert_count = end - last_index;
            if (cur_expert_count <= 0) {
              continue;
            }

            expert_out1.Resize({{cur_expert_count, dim_feedforward}});
            dev_ctx.Alloc<T>(&expert_out1, expert_out1.numel() * sizeof(T));

            Tensor tmp_inp = global_scatter_out.Slice(last_index, end);
            int expert_idx = i * num_expert + idx;
            // linear1   M = cur_expert_count, N = dim_feedforward, K =
            // dim_embed
#ifdef DEBUG_PRINT_LINEAR_SHAPE
            VLOG(0) << "expert id=" << idx
                    << ", liner1 input=" << tmp_inp.dims()
                    << ", weight=" << expert_weights1[expert_idx]->dims()
                    << ", ptr="
                    << (int64_t)(expert_weights1[expert_idx]->data())
                    << ", bias=" << expert_biases1[expert_idx]->dims()
                    << ", ptr=" << (int64_t)(expert_biases1[expert_idx]->data())
                    << ", scale=" << expert_scales1[expert_idx]->dims()
                    << ", ptr=" << (int64_t)(expert_scales1[expert_idx]->data())
                    << ", expert_out1=" << expert_out1.dims();
#endif

            weight_only_gemm.Linear(tmp_inp,
                                    *expert_weights1[expert_idx],
                                    expert_biases1[expert_idx],
                                    *expert_scales1[expert_idx],
                                    cur_expert_count,
                                    dim_feedforward,
                                    dim_embed,
                                    expert_act,  // gelu, relu
                                    &expert_out1);

            // linear2 matmul & add
            Tensor expert_out2 = all_expert_out.Slice(last_index, end);
            // linear2  M = cur_expert_count, N = dim_embed, K = dim_feedforward
#ifdef DEBUG_PRINT_LINEAR_SHAPE
            VLOG(0) << "expert id=" << idx
                    << ", liner2 input=" << expert_out1.dims()
                    << ", weight=" << expert_weights2[expert_idx]->dims()
                    << ", bias=" << expert_biases2[expert_idx]->dims()
                    << ", scale=" << expert_scales2[expert_idx]->dims()
                    << ", expert_out2=" << expert_out2.dims();
            dev_ctx.Wait();
#endif
            weight_only_gemm.Linear(expert_out1,
                                    *expert_weights2[expert_idx],
                                    expert_biases2[expert_idx],
                                    *expert_scales2[expert_idx],
                                    cur_expert_count,
                                    dim_embed,
                                    dim_feedforward,
                                    default_act,  // none
                                    &expert_out2);
#ifdef DEBUG_PRINT_LINEAR_SHAPE
            VLOG(0) << "layer id=" << i << ", expert_idx=" << expert_idx
                    << " end";
            dev_ctx.Wait();
#endif
            last_index = end;
          }
        }
        // at last, concat all expert out
      } else {
        all_expert_out = global_scatter_out;
      }
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      dev_ctx.Wait();
      expert_tm.Pause();
      gather_tm.Resume();
#endif
      // step7. MOEGather
      if (world_size > 1) {
#ifdef DEBUG_PRINT_LINEAR_SHAPE
        VLOG(0) << "layer id=" << i << ", begin gather data all_expert_out=" << all_expert_out.dims() 
          << ", global_gather_out=" << global_gather_out.dims() << ", pos=" << pos.dims();
#endif
        moe_pg.Gather<T>(&all_expert_out, &global_gather_out);
      } else {
        global_gather_out = all_expert_out;
      }
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      dev_ctx.Wait();
      gather_tm.Pause();
#endif
#ifdef DEBUG_PRINT_LINEAR_SHAPE
      VLOG(0) << "layer id=" << i 
        << ", begin  global_gather_out=" << global_gather_out.dims() 
        << ", pos=" << pos.dims()
        << ", moe_gather_out=" << moe_gather_out.dims();
#endif
      // step 7.2, local_gather or scatter
      phi::funcs::GPUScatterAssign<T, int64_t>(
          dev_ctx, global_gather_out, pos, &moe_gather_out, true);

      // step 8, reshape & bmm
      // moe gather out reshape
      moe_gather_out.Resize({{sliced_bsz_seq, topk, dim_embed}});
      topk_value.Resize({{sliced_bsz_seq, 1, topk}});
      phi::BmmKernel<T, phi::GPUContext>(
          dev_ctx, topk_value, moe_gather_out, &bmm_out);
      bmm_out.Resize({{sliced_bsz_seq, dim_embed}});

      // step 9, AllGather
      if (mp_size > 1) {
        // all gather
        moe_pg.AllGather<T>(bmm_out, all_gather_out);
      } else {
        all_gather_out = bmm_out;
      }
      // step 11, add residual
      if (i < layers - 1) {
        // add residual & next layer norm
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
        dev_ctx.Wait();
        ln_tm.Resume();
#endif
        auto *ln_scale_data = ln_scales[i + 1]->data<U>();
        auto *ln_bias_data = ln_biases[i + 1]->data<U>();
        fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
            dev_ctx,
            all_gather_out.data<T>(),        // src
            bias_dropout_residual_out_data,  // residual
            nullptr,                         // bias
            ln_scale_data,
            ln_bias_data,
            moe_out.data<T>(),  // add out, next layer real input, for residual
            dropout_mask_out_data,
            buf0.data<T>(),  // out, after layernorm
            ln_mean_data,
            ln_var_data);
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
        dev_ctx.Wait();
        ln_tm.Pause();
#endif
      } else {
        // last layer, only add residual
        phi::AddKernel<T, phi::GPUContext>(
            dev_ctx, all_gather_out, bias_dropout_residual_out, &moe_out);
      }
      x_data = moe_out.data<T>();
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
      dev_ctx.Wait();
      trans_tm.Pause();
#endif
    }  // layers loop end
    out->Resize({{bsz, seq_len, dim_embed}});
#ifdef DEBUG_TMPROFILE_WEIGHT_ONLY
    dev_ctx.Wait();
    all_tm.Pause();
    VLOG(0) << "gpu=" << static_cast<int>(dev_ctx.GetPlace().GetDeviceId())
            << ", bsz=" << bsz << ", seq_len=" << seq_len
            << ", total span=" << all_tm.ElapsedMS()
            << ", input=" << other_tm.ElapsedMS()
            << ", transformer=" << trans_tm.ElapsedMS()
            << ", [qkv=" << qkv_tm.ElapsedMS()
            << ", fmha=" << fmha_tm.ElapsedMS()
            << ", out_linear=" << out_linear_tm.ElapsedMS()
            << ", expert=" << expert_tm.ElapsedMS()
            << ", ln=" << ln_tm.ElapsedMS()
            << ", gate/all2all=" << gate_tm.ElapsedMS() << "/"
            << gate_nccl_tm.ElapsedMS()
            << ", scatter=" << scatter_tm.ElapsedMS()
            << ", gather=" << gather_tm.ElapsedMS() << "]";
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    fused_multi_transformer_moe_weight_only,
    ops::FusedMultiTransformerMoeWeightOnlyOpKernel<plat::float16>);