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

#include "paddle/fluid/operators/fused/fused_multi_transformer_moe_op.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
// #define _DEBUG_FUSED_MULTI_TRANSFORMER

template <typename T>
class FusedMultiTransformerMoeOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    auto &dev_ctx = ctx.cuda_device_context();

    auto *time_step = ctx.Input<Tensor>("TimeStep");
    // 0. input
    auto *input_x = ctx.Input<Tensor>("X");
    const auto input_x_dims = input_x->dims();
    int bsz = input_x_dims[0];
    int seq_len = input_x_dims[1];
    int dim_embed = input_x_dims[2];
    int bsz_seq = bsz * seq_len;
    if (bsz_seq == 0) {
      return;
    }
    // LOG(INFO) << "intput X: bsz: " << bsz << ", seq_len: " << seq_len << ", dim_embed: " << dim_embed;
    const std::string act_method = ctx.Attr<std::string>("act_method");
    auto *sequence_lengths = ctx.Input<Tensor>("SeqLengths"); // nullptr
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
    auto qkv_biases = ctx.MultiInput<Tensor>("QKVBias");
    const bool trans_qkvw = ctx.Attr<bool>("trans_qkvw");
    const auto qkv_w_dims = qkv_weights[0]->dims();
    int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
    int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];
    int hidden_size = num_head * dim_head;
    int output_size = 3 * hidden_size;
    int input_size = dim_embed;

    bool compute_bias = qkv_biases.size() > 0 && time_step == nullptr;
    // (transA, transB, compute_bias) = (false, trans_qkvw, false)
    auto qkv_compute = AttnMatMul<T>(dev_ctx,
                                     false,
                                     trans_qkvw,
                                     bsz_seq,
                                     output_size,
                                     input_size,
                                     compute_bias);
    Tensor qkv_out;
    qkv_out.Resize({{bsz, seq_len, 3, num_head, dim_head}});
    auto *qkv_out_data =
        dev_ctx.Alloc<T>(&qkv_out, qkv_out.numel() * sizeof(T));

    // 3. fmha
    auto dropout_implementation = ctx.Attr<std::string>("dropout_implementation");
    AttnDropoutParam attn_param(
        true, dropout_implementation, 0.0, true, true, 0, nullptr);
    auto fmha_compute =
        FMHARef<T>(dev_ctx, bsz, seq_len, num_head, dim_head, attn_param);
    auto *src_mask = ctx.Input<Tensor>("SrcMask");
    auto cache_kvs = ctx.MultiInput<Tensor>("CacheKV");
    auto cache_kv_outs = ctx.MultiOutput<Tensor>("CacheKVOut");
    
    int time_step_cpu = 0;
    if (time_step) {
      time_step_cpu = src_mask->dims()[3] - 1;
    }

    auto out_seq_len = seq_len;
    if (time_step) {
      PADDLE_ENFORCE_GT(time_step_cpu,
                        0,
                        platform::errors::PreconditionNotMet(
                            "The value of time_step must > 0, but now is %d",
                            time_step_cpu));
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
    qk_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
    auto *qk_out_data = dev_ctx.Alloc<T>(&qk_out, qk_out.numel() * sizeof(T));

    Tensor softmax_out;
    Tensor attn_dropout_mask_out, attn_dropout_out;
    Tensor qktv_out, fmha_out;
    softmax_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
    auto *softmax_out_data =
        dev_ctx.Alloc<T>(&softmax_out, softmax_out.numel() * sizeof(T));

    qktv_out.Resize({{bsz, num_head, seq_len, dim_head}});
    auto *qktv_out_data =
        dev_ctx.Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
    fmha_out.Resize({{bsz, seq_len, num_head, dim_head}});
    auto *fmha_out_data =
        dev_ctx.Alloc<T>(&fmha_out, fmha_out.numel() * sizeof(T));

    // 4. out_linear
    auto out_linear_weights = ctx.MultiInput<Tensor>("OutLinearW");
    auto out_linear_biases = ctx.MultiInput<Tensor>("OutLinearBias");
    int ring_id = ctx.Attr<int>("ring_id");
    // (transA, transB, compute_bias) = (false, false, false)
    auto out_linear_compute = AttnMatMul<T>(
        dev_ctx, false, false, bsz_seq, dim_embed, hidden_size, false);

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
    auto expert_biases1 = ctx.MultiInput<Tensor>("ExpertBias1");
    auto expert_weights2 = ctx.MultiInput<Tensor>("ExpertWeight2");
    auto expert_biases2 = ctx.MultiInput<Tensor>("ExpertBias2");
    int dim_feedforward = expert_weights1[0]->dims()[1];
    // int dim_feedforward = expert_weights1[0]->dims()[2]; // batched gemm
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
    dev_ctx.Alloc<T>(&topk_idx, topk_idx.numel() * sizeof(T));
    // local expert count, global expert count
    Tensor local_expert_count, global_expert_count;
    local_expert_count.Resize({{tot_expert}});
    global_expert_count.Resize({{tot_expert}});
    dev_ctx.Alloc<int64_t>(&local_expert_count, local_expert_count.numel() * sizeof(int64_t));
    dev_ctx.Alloc<int64_t>(&global_expert_count, global_expert_count.numel() * sizeof(int64_t));
    // fwd_expert_count, fwd_batch_size
    Tensor fwd_expert_count, fwd_batch_size;
    Tensor fwd_expert_count_cpu, fwd_batch_size_cpu;
    fwd_expert_count.Resize({{num_expert}});
    fwd_batch_size.Resize({{1}});
    dev_ctx.Alloc<int64_t>(&fwd_expert_count, fwd_expert_count.numel() * sizeof(int64_t));
    dev_ctx.Alloc<int64_t>(&fwd_batch_size, fwd_batch_size.numel() * sizeof(int64_t));
    // pos, temp pos
    Tensor pos, temp_pos;
    pos.Resize({{out_batch_size}});
    temp_pos.Resize({{out_batch_size}});
    dev_ctx.Alloc<int64_t>(&pos, pos.numel() * sizeof(int64_t));
    dev_ctx.Alloc<int64_t>(&temp_pos, temp_pos.numel() * sizeof(int64_t));
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
    phi::FullKernel<int64_t, phi::GPUContext>(dev_ctx, {1}, topk, pos.dtype(), &topk_tensor);
    // for nccl comm
    auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

    // // expert out, alloc max size
    // Tensor global_scatter_out;
    // global_scatter_out.Resize({{2 * bsz_seq, dim_embed}});
    // dev_ctx.Alloc<T>(&global_scatter_out, global_scatter_out.numel() * sizeof(T));

    // Tensor expert_out1, expert_out2, all_expert_out;
    // expert_out1.Resize({{2 * bsz_seq, dim_feedforward}});
    // // act_bias_out.Resize({{bsz_seq, dim_feedforward}});
    // expert_out2.Resize({{2 * bsz_seq, dim_embed}});
    // all_expert_out.Resize({{2 * bsz_seq, dim_embed}});
    // dev_ctx.Alloc<T>(&expert_out1, expert_out1.numel() * sizeof(T));
    // // dev_ctx.Alloc<T>(&act_bias_out, act_bias_out.numel() * sizeof(T));
    // dev_ctx.Alloc<T>(&expert_out2, expert_out2.numel() * sizeof(T));
    // dev_ctx.Alloc<T>(&all_expert_out, all_expert_out.numel() * sizeof(T));

    Tensor buf0, moe_out;
    buf0.Resize({{bsz_seq, dim_embed}});
    dev_ctx.Alloc<T>(&buf0, buf0.numel() * sizeof(T));
    moe_out.Resize({{bsz_seq, dim_embed}});
    dev_ctx.Alloc<T>(&moe_out, moe_out.numel() * sizeof(T));

    const T *x_data;
    x_data = input_x->data<T>();

    int layers = qkv_weights.size();

    for (int i = 0; i < layers; ++i) {
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step1, pre layernorm";
#endif
      // step1. layer_norm, only layer 0
      if (i == 0) {
        auto *ln_scale_data = ln_scales[i]->data<U>();
        auto *ln_bias_data = ln_biases[i]->data<U>();
        // TODO(wangxi): can remove mean var in inference
        ln_compute.ComputeForward(x_data,
                                  ln_scale_data,
                                  ln_bias_data,
                                  buf0.data<T>(),
                                  ln_mean_data,
                                  ln_var_data);
      }
      // auto *ln_scale_data = ln_scales[i]->data<U>();
      // auto *ln_bias_data = ln_biases[i]->data<U>();
      // // TODO(wangxi): can remove mean var in inference
      // ln_compute.ComputeForward(x_data,
      //                           ln_scale_data,
      //                           ln_bias_data,
      //                           buf0.data<T>(),
      //                           ln_mean_data,
      //                           ln_var_data);

      // step2. qkv
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step2, qkv";
#endif
      const Tensor *qkv_bias =
          qkv_biases.size() > 0 ? qkv_biases[i] : nullptr;
      // NOTE: in decoder stage, bias is fused in fmha
      const Tensor *bias = time_step ? nullptr : qkv_bias;
      qkv_compute.ComputeForward(
        qkv_weights[i], &buf0, bias, &qkv_out, &qkv_out);

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step3.1 fmha";
#endif
      // step3. fmha
      const Tensor *cache_kv =
          cache_kvs.size() > 0 ? cache_kvs[i] : nullptr;
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
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step3.2 out linear";
#endif
      // 输出到buf0
      out_linear_compute.ComputeForward(
          out_linear_weights[i], &fmha_out, nullptr, &buf0, nullptr);
      AllReduce<T>(buf0, ring_id, buf0.numel(), dev_ctx);

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step4";
#endif

      // step5. ln(residual + dropout(input + bias))，在MHA里的
      auto *ln_scale_data = ffn_ln_scales[i]->data<U>();
      auto *ln_bias_data = ffn_ln_biases[i]->data<U>();
      auto *out_linear_bias_data = out_linear_biases[i]->data<T>(); 

      // pre layer norm : bias_dropout_residual_out is residual
      fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
          dev_ctx,
          buf0.data<T>(),
          x_data, // residual, moe out
          out_linear_bias_data,
          ln_scale_data,
          ln_bias_data,
          bias_dropout_residual_out_data,
          dropout_mask_out_data,
          buf0.data<T>(), // output to buf0
          ln_mean_data,
          ln_var_data);
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step5";
#endif
      // moe
      // step2 resize and slice ln_out
      if (mp_size > 1) {
        sliced_inp = buf0.Slice(start, end);
      } else {
        sliced_inp = buf0;
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, gate & topk";
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
                                          topk, // scalar
                                          -1, 
                                          true, 
                                          false, 
                                          &topk_value, 
                                          &topk_idx);
      // step4 prepare forward
      // step4.1 number count
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, number count";
#endif
      phi::NumberCountKernel<int64_t, phi::GPUContext>(dev_ctx, topk_idx, tot_expert, &local_expert_count);
      // step4.2 all_to_all
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, all_to_all";
#endif
      if (world_size > 1) {
        phi::AllToAll<int64_t>(local_expert_count, global_expert_count, moe_ring_id, dev_ctx);
      } else {
        global_expert_count = local_expert_count;
      }

      // global expert count resize
      global_expert_count.Resize({{world_size, num_expert}});
      // fwd expert count
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, fwd expert count";
#endif
      phi::SumKernel<int64_t, phi::GPUContext>(dev_ctx, 
                                               global_expert_count, 
                                               phi::IntArray({0}),
                                               global_expert_count.dtype(),
                                               false,
                                               &fwd_expert_count);
      // fwd batch size
      phi::SumKernel<int64_t, phi::GPUContext>(dev_ctx, 
                                               fwd_expert_count, 
                                               phi::IntArray({}), // axis is None
                                               fwd_expert_count.dtype(),
                                               false,
                                               &fwd_batch_size);
      // step4.3 cumsum & assign pos
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, cumsum";
#endif
      phi::CumsumKernel<int64_t, phi::GPUContext>(dev_ctx, 
                                                  local_expert_count, 
                                                  0,
                                                  false,
                                                  false,
                                                  false,
                                                  &lec_cum);
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, assign pos";
#endif
      phi::AssignPosCompute<int64_t>(dev_ctx, &lec_cum, &topk_idx, &pos, out_batch_size);
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, floor divide";
#endif
      if (topk > 1) {
        phi::FloorDivideKernel<int64_t, phi::GPUContext>(dev_ctx,
                                                         pos,
                                                         topk_tensor,
                                                         &temp_pos);
      } else {
        temp_pos = pos;
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, tensor copy";
#endif
      framework::TensorCopySync(fwd_expert_count, platform::CPUPlace(), &fwd_expert_count_cpu);
      framework::TensorCopySync(fwd_batch_size, platform::CPUPlace(), &fwd_batch_size_cpu);
      int fwd_bsz = fwd_batch_size_cpu.data<int64_t>()[0];

      Tensor global_scatter_out;
      global_scatter_out.Resize({{fwd_bsz, dim_embed}});
      dev_ctx.Alloc<T>(&global_scatter_out, global_scatter_out.numel() * sizeof(T));

      Tensor all_expert_out;
      all_expert_out.Resize({{fwd_bsz, dim_embed}});
      dev_ctx.Alloc<T>(&all_expert_out, all_expert_out.numel() * sizeof(T));

      // global_scatter_out.Resize({{fwd_bsz, dim_embed}});
      // all_expert_out.Resize({{fwd_bsz, dim_embed}});

      // step 5, MOEScatter
      // step 5.1, index select
      // suppose tmp_pos->shape != [0]
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, index select";
#endif
      phi::IndexSelectKernel<T, phi::GPUContext>(dev_ctx, sliced_inp, temp_pos, 0, &index_select_out);
      if (world_size > 1) {
        // auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
        // step 5.2, global_scatter
        if (map->has(moe_ring_id)) {
          phi::GlobalScatterProcessGroupFunctor<T>(dev_ctx, 
                                                   &index_select_out, 
                                                   &local_expert_count, 
                                                   &global_expert_count, 
                                                   moe_ring_id,
                                                   true,
                                                   &global_scatter_out);
        } else {
          phi::GlobalScatterFunctor<T>(dev_ctx, 
                                        &index_select_out, 
                                        &local_expert_count, 
                                        &global_expert_count, 
                                        moe_ring_id,
                                        false,
                                        &global_scatter_out);
        }
      } else {
        global_scatter_out = index_select_out;
      }

      // step 6, Expert Computation
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, Expert Computation";
#endif
      if (fwd_bsz != 0) {
        phi::funcs::ConcatFunctor<phi::GPUContext, T> concat;
        std::vector<Tensor> tmp_expert_out;
        // if (time_step) {
        //   // decoder, use batched gemm
        //   Tensor expert_out1, expert_out2;
        //   expert_out1.Resize({{num_expert, fwd_bsz, dim_feedforward}});
        //   expert_out2.Resize({{num_expert, fwd_bsz, dim_embed}});
        //   dev_ctx.Alloc<T>(&expert_out1, expert_out1.numel() * sizeof(T));
        //   dev_ctx.Alloc<T>(&expert_out2, expert_out2.numel() * sizeof(T));

        //   BatchedMatMulAndAdd<T>(dev_ctx,
        //                          expert_weights1[i],
        //                          &global_scatter_out,
        //                          expert_biases1[i], // bias
        //                          false,
        //                          false,
        //                          true, // compute bias
        //                          true, // is linear1
        //                          &expert_out1,
        //                          &expert_out1); // bias out
        //   phi::GeluKernel<T, phi::GPUContext>(dev_ctx, expert_out1, approximate, &expert_out1);
        //   BatchedMatMulAndAdd<T>(dev_ctx,
        //                          expert_weights2[i],
        //                          &expert_out1, // input
        //                          expert_biases2[i],
        //                          false,
        //                          false,
        //                          true, // compute bias
        //                          false, // is linear1
        //                          &expert_out2,
        //                          &expert_out2);
        //   int last_index = 0;
        //   for (int idx = 0; idx < num_expert; idx++) {
        //     int cur_expert_count = fwd_expert_count_cpu.data<int64_t>()[idx];
        //     if (cur_expert_count <= 0) {
        //       continue;
        //     }
        //     int end = cur_expert_count + last_index;
        //     // expert_out2 slice
        //     Tensor tmp_sliced;
        //     phi::SliceCompute<T, phi::GPUContext, 3>(dev_ctx,
        //                                              expert_out2,
        //                                              {0, 1},
        //                                              {idx, last_index},
        //                                              {idx + 1, end},
        //                                              {1, 1},
        //                                              {},
        //                                              &tmp_sliced);
        //     tmp_sliced.Resize({{cur_expert_count, dim_embed}}); // maybe dont need resize
        //     tmp_expert_out.emplace_back(tmp_sliced);
        //     last_index = end;
        //   }
        // } else {

        // encoder, use matmul
        int last_index = 0;
        // std::vector<Tensor> tmp_expert_out;
        for (int idx = 0; idx < num_expert; idx++) {
          int cur_expert_count = fwd_expert_count_cpu.data<int64_t>()[idx];
          if (cur_expert_count <= 0) {
            continue;
          }
          int end = cur_expert_count + last_index;

          Tensor expert_out1, expert_out2, act_bias_out;
          expert_out1.Resize({{cur_expert_count, dim_feedforward}});
          expert_out2.Resize({{cur_expert_count, dim_embed}});
          act_bias_out.Resize({{cur_expert_count, dim_feedforward}});
          dev_ctx.Alloc<T>(&expert_out1, expert_out1.numel() * sizeof(T));
          dev_ctx.Alloc<T>(&expert_out2, expert_out2.numel() * sizeof(T));
          dev_ctx.Alloc<T>(&act_bias_out, act_bias_out.numel() * sizeof(T));

          FusedDropoutHelper<T, uint8_t> fused_act_dropout_helper(
            dev_ctx, cur_expert_count, dim_feedforward, dropout_param);
          
          Tensor tmp_inp = global_scatter_out.Slice(last_index, end);
          int expert_idx = i * num_expert + idx;

          // linear1 matmul
          // VLOG(0) << "moe, Expert Computation, linear1 mul";
          phi::MatMulAndAdd<T>(dev_ctx, 
                               expert_weights1[expert_idx], 
                               &tmp_inp, 
                               nullptr, 
                               false,
                               false,
                               false,  // dont compute bias
                               &expert_out1, 
                               nullptr);
          // MatMulAndAdd<T>(dev_ctx, 
          //                 expert_weights1[i]->data<T>() + idx * dim_embed * dim_feedforward,
          //                 tmp_inp.data<T>(), 
          //                 nullptr, // bias
          //                 cur_expert_count,
          //                 dim_feedforward,
          //                 dim_embed,
          //                 false,
          //                 false,
          //                 false,  // dont compute bias
          //                 expert_out1.data<T>(), 
          //                 nullptr);

          // bias gelu
          // VLOG(0) << "moe, Expert Computation, add bias & gelu";
          // inplace
          fused_act_dropout_helper.DropoutActBias(dev_ctx,
                                                  expert_out1.data<T>(),
                                                  expert_biases1[expert_idx]->data<T>(),
                                                  "gelu",
                                                  act_bias_out.data<T>(),
                                                  nullptr,
                                                  1.0,
                                                  nullptr,
                                                  0,
                                                  1.0,
                                                  1,
                                                  127.0,
                                                  -127.0,
                                                  approximate);
          // fused_act_dropout_helper.DropoutActBias(dev_ctx,
          //                                         expert_out1.data<T>(),
          //                                         expert_biases1[i]->data<T>() + idx * dim_feedforward,
          //                                         "gelu",
          //                                         act_bias_out.data<T>(),
          //                                         nullptr,
          //                                         1.0,
          //                                         nullptr,
          //                                         0,
          //                                         1.0,
          //                                         1,
          //                                         127.0,
          //                                         -127.0,
          //                                         approximate);

          // linear2 matmul & add
          // VLOG(0) << "moe, Expert Computation, linear2 matmul & add";
          phi::MatMulAndAdd<T>(dev_ctx, 
                               expert_weights2[expert_idx], 
                               &act_bias_out, 
                               expert_biases2[expert_idx], 
                               false,
                               false,
                               true,  //  compute bias
                               &expert_out2, 
                               &expert_out2);
          // MatMulAndAdd<T>(dev_ctx, 
          //                 expert_weights2[i]->data<T>() + idx * dim_embed * dim_feedforward,
          //                 act_bias_out.data<T>(), 
          //                 expert_biases2[i]->data<T>() + idx * dim_embed, 
          //                 cur_expert_count,
          //                 dim_embed,
          //                 dim_feedforward,
          //                 false,
          //                 false,
          //                 true,  //  compute bias
          //                 expert_out2.data<T>(), 
          //                 expert_out2.data<T>());
          // Addmm<T>(dev_ctx, 
          //          *expert_biases2[expert_idx],
          //          act_bias_out,
          //          *expert_weights2[expert_idx],
          //          1.0,
          //          1.0,
          //          &expert_out2);
          tmp_expert_out.emplace_back(expert_out2);
          last_index = end;
          // }
        }
        // at last, concat all expert out
        concat(dev_ctx, tmp_expert_out, 0, &all_expert_out);
      } else {
        all_expert_out = global_scatter_out;
      }

      // step7. MOEGather
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, MOEGather";
#endif
      if (world_size > 1) {
        // auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
        // step 7.1, global_gather
        if (map->has(moe_ring_id)) {
          phi::GlobalGatherProcessGroupFunctor<T>(dev_ctx, 
                                                  &all_expert_out, 
                                                  &local_expert_count, 
                                                  &global_expert_count, 
                                                  moe_ring_id,
                                                  true,
                                                  &global_gather_out);
        } else {
          phi::GlobalGatherFunctor<T>(dev_ctx, 
                                      &all_expert_out, 
                                      &local_expert_count, 
                                      &global_expert_count, 
                                      moe_ring_id,
                                      false,
                                      &global_gather_out);
        }
      } else {
        global_gather_out = all_expert_out;
      }
      // step 7.2, local_gather or scatter
      // suppose pos->shape != [0]
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, local_gather or scatter";
#endif
      phi::ScatterKernel<T, phi::GPUContext>(dev_ctx, 
                                             moe_gather_out, 
                                             pos, 
                                             global_gather_out, 
                                             true, 
                                             &moe_gather_out);
      // step 8, reshape & bmm
      // moe gather out reshape
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, reshape & bmm";
#endif
      moe_gather_out.Resize({{sliced_bsz_seq, topk, dim_embed}});
      topk_value.Resize({{sliced_bsz_seq, 1, topk}});
      phi::BmmKernel<T, phi::GPUContext>(dev_ctx, topk_value, moe_gather_out, &bmm_out);
      bmm_out.Resize({{sliced_bsz_seq, dim_embed}});
      // step 9, AllGather
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, AllGather";
#endif
      if (mp_size > 1) {
        // all gather
        phi::AllGather<T>(bmm_out, all_gather_out, moe_ring_id, dev_ctx);
      } else {
        all_gather_out = bmm_out;
      }

      // step 11, add residual
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "moe, add residual";
#endif
      if (i < layers - 1) {
        // add residual & next layer norm
        auto *ln_scale_data = ln_scales[i + 1]->data<U>();
        auto *ln_bias_data = ln_biases[i + 1]->data<U>();
        fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
            dev_ctx,
            all_gather_out.data<T>(), // src
            bias_dropout_residual_out_data, // residual
            nullptr, // bias
            ln_scale_data,
            ln_bias_data,
            moe_out.data<T>(), // add out, next layer real input, for residual
            dropout_mask_out_data,
            buf0.data<T>(), // out, after layernorm
            ln_mean_data,
            ln_var_data);
      } else {
        // last layer, only add residual
        phi::AddKernel<T, phi::GPUContext>(dev_ctx, all_gather_out, bias_dropout_residual_out, &moe_out);
      }

      // phi::AddKernel<T, phi::GPUContext>(dev_ctx, all_gather_out, bias_dropout_residual_out, &moe_out);
      x_data = moe_out.data<T>();

    } // layers loop end
    moe_out.Resize({{bsz, seq_len, dim_embed}});
    *out = moe_out;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_multi_transformer_moe,
                        ops::FusedMultiTransformerMoeOpKernel<plat::float16>,
                        ops::FusedMultiTransformerMoeOpKernel<float>);
