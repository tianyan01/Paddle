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

#include "paddle/phi/kernels/gpu/fused_moe_kernel.cu.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"

namespace phi {
using Tensor = DenseTensor;
namespace framework = paddle::framework;
namespace platform = paddle::platform;

template <typename T, typename DeviceContext>
void FusedMoeKernel(const DeviceContext& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& gate_weight,
                    const DenseTensor& gate_bias,
                    const DenseTensor& ln_scale,
                    const DenseTensor& ln_bias,
                    const std::vector<const DenseTensor*>& experts_weight1,
                    const std::vector<const DenseTensor*>& experts_bias1,
                    const std::vector<const DenseTensor*>& experts_weight2,
                    const std::vector<const DenseTensor*>& experts_bias2,
                    bool pre_layer_norm,
                    float ln_epsilon,
                    int topk,
                    int mp_size,
                    int mp_rank,
                    int num_expert,
                    int world_size,
                    int moe_ring_id,
                    bool approximate,
                    DenseTensor* out) {
  using U = paddle::operators::LayerNormParamType<T>;
  // output
  dev_ctx.template Alloc<T>(out);
  // dim
  auto x_dim = x.dims();
  int bsz = x_dim[0];
  int seq_len = x_dim[1];
  int bsz_seq = bsz * seq_len;
  int d_model = x_dim[2];
  int tot_expert = world_size * num_expert;
  int dim_feedforward = experts_weight1[0]->dims()[1];

  // pre_layer_norm
  Tensor ln_mean, ln_variance;
  ln_mean.Resize({{bsz_seq}});
  ln_variance.Resize({{bsz_seq}});
  paddle::operators::FusedDropoutLayerNormHelper<T, uint8_t>
      pre_layernorm_helper(bsz_seq, d_model, ln_epsilon);

  // tmp out
  Tensor ln_out;
  ln_out.Resize({{bsz, seq_len, d_model}});
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
  // gate linear
  Tensor gate_out;
  gate_out.Resize({{sliced_bsz_seq, tot_expert}});
  dev_ctx.template Alloc<T>(&gate_out);
  // topk
  Tensor topk_value, topk_idx;
  topk_value.Resize({{sliced_bsz_seq, topk}});
  dev_ctx.template Alloc<T>(&topk_value);
  topk_idx.Resize({{sliced_bsz_seq, topk}});
  dev_ctx.template Alloc<int64_t>(&topk_idx);
  // local expert count, global expert count
  Tensor local_expert_count, global_expert_count;
  local_expert_count.Resize({{tot_expert}});
  global_expert_count.Resize({{tot_expert}});
  dev_ctx.template Alloc<int64_t>(&local_expert_count);
  dev_ctx.template Alloc<int64_t>(&global_expert_count);
  // fwd_expert_count, fwd_batch_size
  Tensor fwd_expert_count, fwd_batch_size;
  fwd_expert_count.Resize({{num_expert}});
  fwd_batch_size.Resize({{1}});
  dev_ctx.template Alloc<int64_t>(&fwd_expert_count);
  dev_ctx.template Alloc<int64_t>(&fwd_batch_size);
  // pos, temp pos
  Tensor pos, temp_pos;
  pos.Resize({{out_batch_size}});
  temp_pos.Resize({{out_batch_size}});
  dev_ctx.template Alloc<int64_t>(&pos);
  dev_ctx.template Alloc<int64_t>(&temp_pos);
  // cumsum
  Tensor lec_cum;
  lec_cum.Resize({{tot_expert}});
  dev_ctx.template Alloc<int64_t>(&lec_cum);
  // fused moe ffn tmp out
  Tensor index_select_out;
  index_select_out.Resize({{out_batch_size, d_model}});
  dev_ctx.template Alloc<T>(&index_select_out);
  paddle::operators::DropoutParam dropout_param(
      false, 0, true, true, 0.0, nullptr, 0);

  // step1 layer norm
  if (pre_layer_norm) {
    auto* ln_mean_data = dev_ctx.template Alloc<U>(&ln_mean);
    auto* ln_variance_data = dev_ctx.template Alloc<U>(&ln_variance);
    auto* ln_out_data = dev_ctx.template Alloc<T>(&ln_out);

    const U* ln_scale_ptr = ln_scale.data<U>();
    const U* ln_bias_ptr = ln_bias.data<U>();
    pre_layernorm_helper.LayerNorm(dev_ctx,
                                   x.data<T>(),
                                   ln_scale_ptr,
                                   ln_bias_ptr,
                                   ln_out_data,
                                   ln_mean_data,
                                   ln_variance_data);
  } else {
    ln_out = x;
  }
  // step2 resize and slice ln_out
  ln_out.Resize({{bsz_seq, d_model}});
  if (mp_size > 1) {
    sliced_inp = ln_out.Slice(start, end);
  } else {
    sliced_inp = ln_out;
  }
  // step3 gate & topk
  MatMulAndAdd<T>(dev_ctx,
                  &gate_weight,
                  &sliced_inp,
                  &gate_bias,
                  false,
                  false,
                  true,  //  compute bias
                  &gate_out,
                  &gate_out);
  TopkKernel<T, DeviceContext>(
      dev_ctx, gate_out, Scalar(topk), -1, true, false, &topk_value, &topk_idx);
  // step4 prepare forward
  // step4.1 number count
  // VLOG(0) << "moe, number count";
  NumberCountKernel<int64_t, DeviceContext>(
      dev_ctx, topk_idx, tot_expert, &local_expert_count);
  // step4.2 all_to_all
  // VLOG(0) << "moe, all_to_all";
  if (world_size > 1) {
    AllToAll<int64_t>(
        local_expert_count, global_expert_count, moe_ring_id, dev_ctx);
  } else {
    global_expert_count = local_expert_count;
  }
  // global expert count resize
  global_expert_count.Resize({{world_size, num_expert}});
  // fwd expert count
  // VLOG(0) << "moe, fwd expert count";
  SumKernel<int64_t, DeviceContext>(dev_ctx,
                                    global_expert_count,
                                    IntArray({0}),
                                    global_expert_count.dtype(),
                                    false,
                                    &fwd_expert_count);
  // fwd batch size
  // VLOG(0) << "moe, fwd batch size";
  SumKernel<int64_t, DeviceContext>(dev_ctx,
                                    fwd_expert_count,
                                    IntArray({}),  // axis is None
                                    fwd_expert_count.dtype(),
                                    false,
                                    &fwd_batch_size);
  // step4.3 cumsum & assign pos
  // VLOG(0) << "moe, cumsum & assign pos";
  CumsumKernel<int64_t, DeviceContext>(
      dev_ctx, local_expert_count, Scalar(0), false, false, false, &lec_cum);
  AssignPosCompute<int64_t>(dev_ctx, &lec_cum, &topk_idx, &pos, out_batch_size);
  if (topk > 1) {
    Tensor topk_tensor;
    topk_tensor.Resize({{1}});
    dev_ctx.template Alloc<int64_t>(&topk_tensor);
    FullKernel<int64_t, DeviceContext>(
        dev_ctx, {1}, topk, pos.dtype(), &topk_tensor);
    FloorDivideKernel<int64_t, DeviceContext>(
        dev_ctx, pos, topk_tensor, &temp_pos);
  } else {
    temp_pos = pos;
  }

  Tensor fwd_expert_count_cpu;
  framework::TensorCopy(
      fwd_expert_count, platform::CPUPlace(), &fwd_expert_count_cpu);

  Tensor fwd_batch_size_cpu;
  framework::TensorCopy(
      fwd_batch_size, platform::CPUPlace(), &fwd_batch_size_cpu);
  dev_ctx.Wait();

  int fwd_bsz = fwd_batch_size_cpu.data<int64_t>()[0];

  Tensor global_scatter_out;

  // step 5, MOEScatter
  // step 5.1, index select
  // suppose tmp_pos->shape != [0]
  IndexSelectKernel<T, DeviceContext>(
      dev_ctx, sliced_inp, temp_pos, 0, &index_select_out);
  if (world_size > 1) {
    global_scatter_out.Resize({{fwd_bsz, d_model}});
    dev_ctx.template Alloc<T>(&global_scatter_out);
    auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
    // step 5.2, global_scatter
    if (map->has(moe_ring_id)) {
      GlobalScatterProcessGroupFunctor<T>(dev_ctx,
                                          &index_select_out,
                                          &local_expert_count,
                                          &global_expert_count,
                                          moe_ring_id,
                                          true,
                                          &global_scatter_out);
    } else {
      GlobalScatterFunctor<T>(dev_ctx,
                              &index_select_out,
                              &local_expert_count,
                              &global_expert_count,
                              moe_ring_id,
                              true,
                              &global_scatter_out);
    }
  } else {
    global_scatter_out = index_select_out;
  }

  // step 6, Expert Computation
  Tensor all_expert_out;
  if (fwd_bsz != 0) {
    all_expert_out.Resize({{fwd_bsz, d_model}});
    dev_ctx.template Alloc<T>(&all_expert_out);

    int last_index = 0;
    for (int idx = 0; idx < num_expert; idx++) {
      int cur_expert_count = fwd_expert_count_cpu.data<int64_t>()[idx];
      if (cur_expert_count <= 0) {
        continue;
      }
      int end = cur_expert_count + last_index;

      Tensor expert_out1;
      expert_out1.Resize({{cur_expert_count, dim_feedforward}});
      dev_ctx.template Alloc<T>(&expert_out1);

      paddle::operators::FusedDropoutHelper<T, uint8_t>
          fused_act_dropout_helper(
              dev_ctx, cur_expert_count, dim_feedforward, dropout_param);

      Tensor tmp_inp = global_scatter_out.Slice(last_index, end);
      // linear1 matmul
      MatMulAndAdd<T>(dev_ctx,
                      experts_weight1[idx],
                      &tmp_inp,
                      nullptr,
                      false,
                      false,
                      false,  // dont compute bias
                      &expert_out1,
                      nullptr);
      // bias gelu
      fused_act_dropout_helper.DropoutActBias(dev_ctx,
                                              expert_out1.data<T>(),
                                              experts_bias1[idx]->data<T>(),
                                              "gelu",
                                              expert_out1.data<T>(),
                                              nullptr,
                                              1.0,
                                              nullptr,
                                              0,
                                              1.0,
                                              1,
                                              127.0,
                                              -127.0,
                                              approximate);

      Tensor expert_out2 = all_expert_out.Slice(last_index, end);
      //      expert_out2.Resize({{cur_expert_count, d_model}});
      // linear2 matmul & add
      MatMulAndAdd<T>(dev_ctx,
                      experts_weight2[idx],
                      &expert_out1,
                      experts_bias2[idx],
                      false,
                      false,
                      true,  //  compute bias
                      &expert_out2,
                      &expert_out2);
      last_index = end;
    }
  } else {
    all_expert_out = global_scatter_out;
  }
  // step7. MOEGather
  Tensor global_gather_out;
  if (world_size > 1) {
    global_gather_out.Resize({{out_batch_size, d_model}});
    dev_ctx.template Alloc<T>(&global_gather_out);
    auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
    // step 7.1, global_gather
    if (map->has(moe_ring_id)) {
      GlobalGatherProcessGroupFunctor<T>(dev_ctx,
                                         &all_expert_out,
                                         &local_expert_count,
                                         &global_expert_count,
                                         moe_ring_id,
                                         true,
                                         &global_gather_out);
    } else {
      GlobalGatherFunctor<T>(dev_ctx,
                             &all_expert_out,
                             &local_expert_count,
                             &global_expert_count,
                             moe_ring_id,
                             true,
                             &global_gather_out);
    }
  } else {
    global_gather_out = all_expert_out;
  }
  // step 7.2, local_gather or scatter
  // suppose pos->shape != [0]
  Tensor moe_gather_out;
  if (pre_layer_norm) {
    moe_gather_out.ShareDataWith(index_select_out);
    moe_gather_out.Resize({{out_batch_size, d_model}});
  } else {
    moe_gather_out.Resize({{out_batch_size, d_model}});
    dev_ctx.template Alloc<T>(&moe_gather_out);
  }
  phi::funcs::GPUScatterAssign<T, int64_t>(
      dev_ctx, global_gather_out, pos, &moe_gather_out, true);

  // step 8, reshape & bmm
  // moe gather out reshape
  Tensor bmm_out;
  moe_gather_out.Resize({{sliced_bsz_seq, topk, d_model}});
  topk_value.Resize({{sliced_bsz_seq, 1, topk}});
  bmm_out.ShareDataWith(*out);
  bmm_out.Resize({{sliced_bsz_seq, 1, d_model}});
  //	dev_ctx.template Alloc<T>(&bmm_out);
  BmmKernel<T, DeviceContext>(dev_ctx, topk_value, moe_gather_out, &bmm_out);
  bmm_out.Resize({{sliced_bsz_seq, d_model}});

  Tensor all_gather_out;
  // step 9, AllGather
  if (mp_size > 1) {
    // all gather
    all_gather_out.Resize({{bsz_seq, d_model}});
    dev_ctx.template Alloc<T>(&all_gather_out);
    AllGather<T>(bmm_out, all_gather_out, moe_ring_id, dev_ctx);
  } else {
    all_gather_out = bmm_out;
  }
  // step 10, reshape
  all_gather_out.Resize(x_dim);
  out->Resize(x_dim);
  // step 11, add residual
  // VLOG(0) << "moe, add residual";
  AddKernel<T, DeviceContext>(dev_ctx, all_gather_out, x, out);
  // layer norm
  if (!pre_layer_norm) {
    auto* ln_mean_data = dev_ctx.template Alloc<U>(&ln_mean);
	auto* ln_variance_data = dev_ctx.template Alloc<U>(&ln_variance);
	auto* ln_out_data = dev_ctx.template Alloc<T>(&ln_out);

	const U* ln_scale_ptr = ln_scale.data<U>();
	const U* ln_bias_ptr = ln_bias.data<U>();
    pre_layernorm_helper.LayerNorm(dev_ctx,
                                   out->data<T>(),
                                   ln_scale_ptr,
                                   ln_bias_ptr,
                                   out->data<T>(),
                                   ln_mean_data,
                                   ln_variance_data);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(fused_moe_kernel,
                   GPU,
                   ALL_LAYOUT,
                   phi::FusedMoeKernel,
                   float,
                   double,
                   paddle::platform::float16) {}
