/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_moe_op.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename T>
static void AllToAll(Tensor& tensor,  // NOLINT
                     Tensor& out,
                     const int ring_id,
                     const phi::GPUContext& ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  if (map->has(ring_id)) {
    paddle::distributed::ProcessGroup* pg = map->get(ring_id);
    auto pg_nccl = static_cast<distributed::ProcessGroupNCCL*>(pg);

    std::vector<phi::DenseTensor> in_tensor;
    std::vector<phi::DenseTensor> out_tensor;
    in_tensor.push_back(tensor);
    out_tensor.push_back(out);
    auto task = pg_nccl->AllToAll(in_tensor, out_tensor, true, true);
    task->Wait();
  } else {
    auto dtype = platform::ToNCCLDataType(
        framework::TransToProtoVarType(tensor.dtype()));
    int64_t send_numel = tensor.numel(); // send_numel
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    int nranks = comm->nranks();
    auto stream = ctx.stream();

    framework::DDim x_dims = tensor.dims();
    framework::DDim out_dims(x_dims);
    PADDLE_ENFORCE_EQ(
        x_dims[0] % nranks,
        0,
        platform::errors::InvalidArgument(
            "The first dimension size (%d) of the input tensor must be "
            "divisible by the number of ranks (%d).",
            x_dims[0],
            nranks));
    auto send_buf = tensor.data<T>();
    auto recv_buf = out.mutable_data<T>(out_dims, place);
    size_t offset = 0;
    send_numel /= nranks;
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
    for (auto i = 0; i < nranks; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
          send_buf + offset, send_numel, dtype, i, comm->comm(), stream));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
          recv_buf + offset, send_numel, dtype, i, comm->comm(), stream));
      offset += send_numel;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  }
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

template <typename T>
static void AllGather(Tensor& tensor,  // NOLINT
                      Tensor& out,
                      const int ring_id,
                      const phi::GPUContext& ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  if (map->has(ring_id)) {
    paddle::distributed::ProcessGroup* pg = map->get(ring_id);
    auto pg_nccl = static_cast<distributed::ProcessGroupNCCL*>(pg);

    std::vector<Tensor> in_tensor;
    std::vector<Tensor> out_tensor;
    in_tensor.push_back(tensor);
    out_tensor.push_back(out);
    auto task = pg_nccl->AllGather(in_tensor, out_tensor, true, true);
    task->Wait();
  } else {
    auto dtype = platform::ToNCCLDataType(
        framework::TransToProtoVarType(tensor.dtype()));
    int64_t numel = tensor.numel();
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    auto stream = ctx.stream();
    auto out_dims = tensor.dims();
    int nranks = comm->nranks();
    out_dims[0] *= nranks;
    out.mutable_data<T>(out_dims, place);
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
        tensor.data<T>(), out.data<T>(), numel, dtype, comm->comm(), stream));
  }
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

template <typename DeviceContext, typename T>
class FusedMoeOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using U = LayerNormParamType<T>;
    auto& dev_ctx = context.cuda_device_context();
    // input
    auto* x = context.Input<Tensor>("X");
    auto* gate_weight = context.Input<Tensor>("GateWeight");
    auto* gate_bias = context.Input<Tensor>("GateBias");
    const bool pre_layer_norm = context.Attr<bool>("pre_layer_norm");
    auto* ln_scale = 
        pre_layer_norm ? context.Input<Tensor>("LnScale") : nullptr;
    auto* ln_bias = 
        pre_layer_norm ? context.Input<Tensor>("LnBias") : nullptr;
    // linear 1
    auto experts_weight1 = context.MultiInput<Tensor>("ExpertsWeight1");
    auto experts_bias1 = context.MultiInput<Tensor>("ExpertsBias1");
    // linear 2
    auto experts_weight2 = context.MultiInput<Tensor>("ExpertsWeight2");
    auto experts_bias2 = context.MultiInput<Tensor>("ExpertsBias2");

    // output
    auto* out = context.Output<Tensor>("Out");
    dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));

    // attr
    const float epsilon = context.Attr<float>("ln_epsilon");
    const int topk = context.Attr<int>("topk");
    const int mp_size = context.Attr<int>("mp_size");
    const int mp_rank = context.Attr<int>("mp_rank");
    const int num_expert = context.Attr<int>("num_expert");
    const int world_size = context.Attr<int>("world_size");
    const int moe_ring_id = context.Attr<int>("moe_ring_id");

    // dim
    auto x_dim = x->dims();
    int bsz = x_dim[0];
    int seq_len = x_dim[1];
    int bsz_seq = bsz * seq_len;
    int d_model = x_dim[2];
    int tot_expert = world_size * num_expert;
    int dim_feedforward = experts_weight1[0]->dims()[1];

    // pre_layer_norm
    const U* ln_scale_ptr =
        ln_scale == nullptr ? nullptr : ln_scale->data<U>();
    const U* ln_bias_ptr = 
        ln_bias == nullptr ? nullptr : ln_bias->data<U>();
    Tensor ln_mean, ln_variance;
    ln_mean.Resize({{bsz_seq}});
    auto* ln_mean_data =
        dev_ctx.Alloc<U>(&ln_mean, ln_mean.numel() * sizeof(U));
    ln_variance.Resize({{bsz_seq}});
    auto* ln_variance_data = 
        dev_ctx.Alloc<U>(&ln_variance, ln_variance.numel() * sizeof(U));
    FusedDropoutLayerNormHelper<T, uint8_t> pre_layernorm_helper(
        bsz_seq, d_model, epsilon);
    // tmp out
    Tensor ln_out;
    ln_out.Resize({{bsz, seq_len, d_model}});
    auto *ln_out_data = dev_ctx.Alloc<T>(&ln_out, ln_out.numel() * sizeof(T));
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
    sliced_inp.Resize({{sliced_bsz_seq, d_model}});
    auto* sliced_inp_data = dev_ctx.Alloc<T>(&sliced_inp, sliced_inp.numel() * sizeof(T));
    // gate linear
    Tensor gate_out;
    gate_out.Resize({{sliced_bsz_seq, tot_expert}});
    auto* gate_out_data = dev_ctx.Alloc<T>(&gate_out, gate_out.numel() * sizeof(T));
    auto gate_linear_compute = AttnMatMul<T>(
        dev_ctx, false, false, sliced_bsz_seq, tot_expert, d_model, true);
    // topk
    Tensor topk_value, topk_idx;
    topk_value.Resize({{sliced_bsz_seq, topk}});
    auto* topk_value_data = dev_ctx.Alloc<T>(&topk_value, topk_value.numel() * sizeof(T));
    topk_idx.Resize({{sliced_bsz_seq, topk}});
    auto* topk_idx_data = dev_ctx.Alloc<int64_t>(&topk_idx, topk_idx.numel() * sizeof(int64_t));
    // local expert count, global expert count
    Tensor local_expert_count, global_expert_count;
    local_expert_count.Resize({{tot_expert}});
    global_expert_count.Resize({{tot_expert}});
    auto* local_expert_count_data =  
        dev_ctx.Alloc<int64_t>(&local_expert_count, local_expert_count.numel() * sizeof(int64_t));
    auto* global_expert_count_data =  
        dev_ctx.Alloc<int64_t>(&global_expert_count, global_expert_count.numel() * sizeof(int64_t));
    // fwd_expert_count, fwd_batch_size
    Tensor fwd_expert_count, fwd_batch_size;
    fwd_expert_count.Resize({{world_size, num_expert}});
    fwd_batch_size.Resize({{1}});
    auto* fwd_expert_count_data = 
        dev_ctx.Alloc<int64_t>(&fwd_expert_count, fwd_expert_count.numel() * sizeof(int64_t));
    auto* fwd_batch_size_data = 
        dev_ctx.Alloc<int64_t>(&fwd_batch_size, fwd_batch_size.numel() * sizeof(int64_t));
    // pos, temp pos
    Tensor pos, temp_pos;
    pos.Resize({{out_batch_size}});
    temp_pos.Resize({{out_batch_size}});
    auto* pos_data = dev_ctx.Alloc<int64_t>(&pos, pos.numel() * sizeof(int64_t));
    auto* temp_pos_data = dev_ctx.Alloc<int64_t>(&temp_pos, temp_pos.numel() * sizeof(int64_t));
    // cumsum
    Tensor lec_cum;
    lec_cum.Resize({{tot_expert}});
    auto* lec_cum_data = dev_ctx.Alloc<int64_t>(&lec_cum, lec_cum.numel() * sizeof(int64_t));
    // fused moe ffn tmp out
    Tensor index_select_out;
    index_select_out.Resize({{out_batch_size, d_model}});
    auto* index_select_out_data = dev_ctx.Alloc<T>(&index_select_out, 
                                                   index_select_out.numel() * sizeof(T));
    Tensor global_gather_out;
    global_gather_out.Resize({{out_batch_size, d_model}});
    auto* global_gather_out_data = dev_ctx.Alloc<T>(&global_gather_out, 
                                                    global_gather_out.numel() * sizeof(T));
    Tensor moe_gather_out;
    moe_gather_out.Resize({{out_batch_size, d_model}});
    auto* moe_gather_out_data = dev_ctx.Alloc<T>(&moe_gather_out, 
                                                 moe_gather_out.numel() * sizeof(T));
    Tensor bmm_out;
    bmm_out.Resize({{sliced_bsz_seq, 1, d_model}});
    auto* bmm_out_data = dev_ctx.Alloc<T>(&bmm_out, bmm_out.numel() * sizeof(T));
    Tensor all_gather_out;
    all_gather_out.Resize({{bsz_seq, d_model}});
    auto* all_gather_out_data = 
        dev_ctx.Alloc<T>(&all_gather_out, all_gather_out.numel() * sizeof(T));
    DropoutParam dropout_param(false, 0, true, true, 0.0, nullptr, 0);

    // step1 layer norm
    if (pre_layer_norm) {
      pre_layernorm_helper.LayerNorm(dev_ctx,
                                     x->data<T>(),
                                     ln_scale_ptr,
                                     ln_bias_ptr,
                                     ln_out_data,
                                     ln_mean_data,
                                     ln_variance_data);
    } else {
      ln_out = *x;
    }
    // step2 resize and slice ln_out
    ln_out.Resize({{bsz_seq, d_model}});
    if (mp_size > 1) {
      sliced_inp = ln_out.Slice(start, end);
    } else {
      sliced_inp = ln_out;
    }
    // step3 gate & topk
    gate_linear_compute.ComputeForward(gate_weight, &sliced_inp, gate_bias, &gate_out, &gate_out);
    phi::TopkKernel<T, DeviceContext>(dev_ctx, 
                                      gate_out, 
                                      phi::Scalar(topk), 
                                      -1, 
                                      true, 
                                      false, 
                                      &topk_value, 
                                      &topk_idx);
    // step4 prepare forward
    // step4.1 number count
    NumberCountCompute<int64_t>(dev_ctx, &topk_idx, tot_expert, &local_expert_count);
    // step4.2 all_to_all
    if (world_size > 1) {
      AllToAll<int64_t>(local_expert_count, global_expert_count, moe_ring_id, dev_ctx);
    } else {
      global_expert_count = local_expert_count;
    }
    // global expert count resize
    global_expert_count.Resize({{world_size, num_expert}});
    // fwd expert count
    phi::SumKernel<int64_t, DeviceContext>(dev_ctx, 
                                           global_expert_count, 
                                           phi::IntArray({0}),
                                           global_expert_count.dtype(),
                                           false,
                                           &fwd_expert_count);
    // fwd batch size
    phi::SumKernel<int64_t, DeviceContext>(dev_ctx, 
                                           fwd_expert_count, 
                                           phi::IntArray({}), // axis is None
                                           fwd_expert_count.dtype(),
                                           false,
                                           &fwd_batch_size);
    // step4.3 cumsum & assign pos
    phi::CumsumKernel<int64_t, DeviceContext>(dev_ctx, 
                                              local_expert_count, 
                                              phi::Scalar(0),
                                              false,
                                              false,
                                              false,
                                              &lec_cum);
    AssignPosCompute<int64_t>(dev_ctx, &lec_cum, &topk_idx, &pos);
    if (topk > 1) {
      Tensor topk_tensor;
      topk_tensor.Resize({{1}});
      auto *topk_tensor_data = dev_ctx.Alloc<T>(&topk_tensor, topk_tensor.numel() * sizeof(int64_t));
      phi::FullKernel<int64_t, DeviceContext>(dev_ctx, {1}, topk, pos.dtype(), &topk_tensor);
      phi::FloorDivideKernel<int64_t, DeviceContext>(dev_ctx,
                                                     pos,
                                                     topk_tensor,
                                                     &temp_pos);
    } else {
      temp_pos = pos;
    }
    Tensor fwd_expert_count_cpu;
    framework::TensorCopySync(fwd_expert_count, platform::CPUPlace(), &fwd_expert_count_cpu);
    Tensor fwd_batch_size_cpu;
    framework::TensorCopySync(fwd_batch_size, platform::CPUPlace(), &fwd_batch_size_cpu);
    int fwd_bsz = fwd_batch_size_cpu.data<int64_t>()[0];

    Tensor global_scatter_out;
    global_scatter_out.Resize({{fwd_bsz, d_model}});
    auto* global_scatter_out_data = dev_ctx.Alloc<T>(&global_scatter_out, 
                                                     global_scatter_out.numel() * sizeof(T));
    std::vector<Tensor> tmp_expert_out;
    Tensor all_expert_out;
    all_expert_out.Resize({{fwd_bsz, d_model}});
    auto* all_expert_out_data = dev_ctx.Alloc<T>(&all_expert_out, 
                                                 all_expert_out.numel() * sizeof(T));
    // step 5, MOEScatter
    // step 5.1, index select
    // suppose tmp_pos->shape != [0]
    phi::IndexSelectKernel<T, DeviceContext>(dev_ctx, sliced_inp, temp_pos, 0, &index_select_out);
    if (world_size > 1) {
      auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
      // step 5.2, global_scatter
      if (map->has(moe_ring_id)) {
        GlobalScatterProcessGroupFunctor<DeviceContext, T> functor_;
        functor_(dev_ctx, 
                 &index_select_out, 
                 &local_expert_count, 
                 &global_expert_count, 
                 moe_ring_id,
                 true,
                 &global_scatter_out);
      } else {
        GlobalScatterFunctor<DeviceContext, T> functor_;
        functor_(dev_ctx, 
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
    if (global_scatter_out.dims()[0] != 0) {
      int last_index = 0;
      for (int idx = 0; idx < num_expert; idx++) {
        int cur_expert_count = fwd_expert_count_cpu.data<int64_t>()[idx];
        if (cur_expert_count <= 0) {
          continue;
        }
        int end = cur_expert_count + last_index;
        Tensor expert_out1;
        expert_out1.Resize({{cur_expert_count, dim_feedforward}});
        auto *expert_out1_data = dev_ctx.Alloc<T>(&expert_out1, 
                                                  expert_out1.numel() * sizeof(T));
        Tensor act_bias_out;
        act_bias_out.Resize({{cur_expert_count, dim_feedforward}});
        auto *act_bias_out_data = dev_ctx.Alloc<T>(&act_bias_out, 
                                                   act_bias_out.numel() * sizeof(T));
        Tensor expert_out2;
        expert_out2.Resize({{cur_expert_count, d_model}});
        auto *expert_out2_data = dev_ctx.Alloc<T>(&expert_out2, 
                                                  expert_out2.numel() * sizeof(T));
        FusedDropoutHelper<T, uint8_t> fused_act_dropout_helper(
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
                                                act_bias_out.data<T>(),
                                                nullptr);
        // linear2 matmul & add
        MatMulAndAdd<T>(dev_ctx, 
                        experts_weight2[idx], 
                        &act_bias_out, 
                        experts_bias2[idx], 
                        false,
                        false,
                        true,  //  compute bias
                        &expert_out2, 
                        &expert_out2);
        tmp_expert_out.emplace_back(expert_out2);
        last_index = end;
      }
      phi::funcs::ConcatFunctor<DeviceContext, T> concat;
      concat(dev_ctx, tmp_expert_out, 0, &all_expert_out);
    } else {
      all_expert_out = global_scatter_out;
    }
    // step7. MOEGather
    if (world_size > 1) {
      auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
      // step 7.1, global_gather
      if (map->has(moe_ring_id)) {
        GlobalGatherProcessGroupFunctor<DeviceContext, T> functor_;
        functor_(dev_ctx, 
                 &all_expert_out, 
                 &local_expert_count, 
                 &global_expert_count, 
                 moe_ring_id,
                 true,
                 &global_gather_out);
      } else {
        GlobalGatherFunctor<DeviceContext, T> functor_;
        functor_(dev_ctx, 
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
    phi::ScatterKernel<T, DeviceContext>(dev_ctx, 
                                         moe_gather_out, 
                                         pos, 
                                         global_gather_out, 
                                         true, 
                                         &moe_gather_out);
    // step 8, reshape & bmm
    if (topk > 1) {
      // moe gather out reshape
      moe_gather_out.Resize({{sliced_bsz_seq, topk, d_model}});
      topk_value.Resize({{sliced_bsz_seq, 1, topk}});
      phi::BmmKernel<T, DeviceContext>(dev_ctx, topk_value, moe_gather_out, &bmm_out);
      bmm_out.Resize({{sliced_bsz_seq, d_model}});
    } else {
      bmm_out = moe_gather_out;
    }
    // step 9, AllGather
    if (mp_size > 1) {
      // all gather
      AllGather<T>(bmm_out, all_gather_out, moe_ring_id, dev_ctx);
    } else {
      all_gather_out = bmm_out;
    }
    // step 10, reshape
    all_gather_out.Resize(x_dim);
    // step 11, add residual
    phi::AddKernel<T, DeviceContext>(dev_ctx, all_gather_out, *x, out);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fused_moe,
    ops::FusedMoeOpKernel<phi::GPUContext, float>,
    ops::FusedMoeOpKernel<phi::GPUContext, double>,
    ops::FusedMoeOpKernel<phi::GPUContext, paddle::platform::float16>);
