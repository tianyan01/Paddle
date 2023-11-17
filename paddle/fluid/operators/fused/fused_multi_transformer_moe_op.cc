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

#include <memory>
#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

class FusedMultiTransformerMoeOp : public framework::OperatorWithKernel {
 private:
  static constexpr const char *OpName = "FusedMultiTransformerMoeOp";

 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
#define CHECK_INPUT(name) \
  OP_INOUT_CHECK(ctx->HasInput(#name), "Input", #name, OpName)
#define CHECK_INPUTS(name) \
  OP_INOUT_CHECK(ctx->HasInputs(#name), "Input", #name, OpName)
#define CHECK_OUTPUT(name) \
  OP_INOUT_CHECK(ctx->HasOutput(#name), "Output", #name, OpName)
#define CHECK_OUTPUTS(name) \
  OP_INOUT_CHECK(ctx->HasOutputs(#name), "Output", #name, OpName)

    CHECK_INPUT(X);

    // attention
    CHECK_INPUTS(QKVW);
    CHECK_INPUTS(OutLinearW);

    if (ctx->HasInput("TimeStep")) {
      CHECK_INPUTS(CacheKV);
    }

    if (ctx->HasInputs("CacheKV")) {
      CHECK_OUTPUTS(CacheKVOut);
    }

    // moe
    CHECK_INPUTS(GateWeight);
    CHECK_INPUTS(GateBias);
    CHECK_INPUTS(ExpertWeight1);
    CHECK_INPUTS(ExpertWeight2);

    // out
    CHECK_OUTPUT(Out);

    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputsDim("QKVW")[0];
    bool trans_qkvw = ctx->Attrs().Get<bool>("trans_qkvw");
    PADDLE_ENFORCE_EQ(
        x_dim.size(),
        3,
        platform::errors::InvalidArgument("The dimensions of x must be 3"
                                          "(batch_size, seq_len, dim_embed),"
                                          "but received dimensions of"
                                          "Input is [%d]",
                                          x_dim.size()));
    PADDLE_ENFORCE_EQ(y_dim.size(),
                      4,
                      platform::errors::InvalidArgument(
                          "The dimensions of qkv_weight must be 4"
                          "(3, num_head, dim_head, dim_embed),"
                          "but received dimensions of"
                          "Input is [%d]",
                          y_dim.size()));
    PADDLE_ENFORCE_EQ(
        x_dim[2],
        trans_qkvw ? y_dim[3] : y_dim[0],
        platform::errors::InvalidArgument(
            "ShapeError: the dimension of x_dim[2] and y_dim[3](trans_qkvw is "
            "true) or y_dim[0](trans_qkvw is false)"
            "must be equal. But received: the shape "
            "of input x = [%s], and the shape of "
            "input qkv_weight = [%s]",
            x_dim,
            y_dim));

    if (ctx->HasInputs("CacheKV")) {
      // [2, batch_size, num_head, max_seq_len, head_size]
      const auto &c_dims = ctx->GetInputsDim("CacheKV");
      const auto &c_dim = c_dims[0];

      PADDLE_ENFORCE_EQ(
          c_dim.size(),
          5,
          paddle::platform::errors::InvalidArgument(
              "The CacheKV must be 5 dims, but got %d", c_dim.size()));
      PADDLE_ENFORCE_EQ(c_dim[0],
                        2,
                        paddle::platform::errors::InvalidArgument(
                            "The first dim of CacheKV must be 2, but got %d",
                            c_dim[0]));  // 2
      PADDLE_ENFORCE_EQ(c_dim[2],
                        trans_qkvw ? y_dim[1] : y_dim[2],
                        paddle::platform::errors::InvalidArgument(
                            "The third dim of CacheKV must be equal with num "
                            "head %d, but got %d",
                            trans_qkvw ? y_dim[1] : y_dim[2],
                            c_dim[2]));  // num_head
      PADDLE_ENFORCE_EQ(c_dim[4],
                        trans_qkvw ? y_dim[2] : y_dim[3],
                        paddle::platform::errors::InvalidArgument(
                            "The fifth dim of CacheKV must be equal with head "
                            "size %d, but got %d",
                            trans_qkvw ? y_dim[2] : y_dim[3],
                            c_dim[4]));  // head_size
    }

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "TimeStep") {
      VLOG(10) << "var_name:" << var_name << " need not to transform";
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

class FusedMultiTransformerMoeOpOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("LnScale",
             "Scale is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDuplicable();
    AddInput("LnBias",
             "Bias is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDuplicable();
    AddInput("QKVW", "The qkv weight tensor.").AsDuplicable();
    AddInput("QKVBias", "The qkv bias tensor.").AsDispensable().AsDuplicable();
    AddInput("CacheKV", "(optional) The cached KV for generation inference.")
        .AsDispensable()
        .AsDuplicable();
    AddInput("PreCaches",
             "(optional) The prefix caches for generation inference.")
        .AsDispensable()
        .AsDuplicable();
    AddInput("RotaryPosEmb",
             "(optional) The RoPE embeddings for generation inference.")
        .AsDispensable();
    AddInput("BeamCacheOffset",
             "(optional) The offset of CacheKV when using BeamSearch.")
        .AsDispensable();
    AddInput("TimeStep",
             "(optional, int) The time step for generation inference.")
        .AsDispensable();
    AddInput("SeqLengths", "(optional) The sequence length tensor of inputs.")
        .AsDispensable();
    AddInput("SrcMask", "(optional) The attention mask tensor in fmha.")
        .AsDispensable();
    AddInput("OutLinearW", "The out_linear weight tensor.").AsDuplicable();
    AddInput("OutLinearBias", "The out_linear bias tensor.")
        .AsDispensable()
        .AsDuplicable();
    AddInput("GateWeight", "The gate_weights in moe")
        .AsDuplicable();
    AddInput("GateBias", "The gate_biases in moe")
        .AsDuplicable();
    AddInput("FFNLnScale", "The layer_norm scale of FusedFeedForward op")
        .AsDuplicable();
    AddInput("FFNLnBias", "The layer_norm bias of FusedFeedForward op")
        .AsDuplicable();
    AddInput("ExpertWeight1", "The expert_weights1 in moe")
        .AsDuplicable();
    AddInput("ExpertBias1", "The expert_biases1 in moe")
        .AsDuplicable();
    AddInput("ExpertWeight2", "The expert_weights2 in moe")
        .AsDuplicable();
    AddInput("ExpertBias2", "The expert_biases2 in moe")
        .AsDuplicable();
    AddOutput("CacheKVOut", "The updated cache KV. Inplace with CacheKV")
        .AsDispensable()
        .AsDuplicable();
    AddOutput("Out", "Result after multi .");
    AddAttr<bool>("pre_layer_norm",
                  "if true, the attention op uses pre_layer_norm architecure, "
                  "else, uses post_layer_norm architecuture. "
                  "[default true].")
        .SetDefault(true);
    AddAttr<float>("epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f,
                            true,
                            platform::errors::InvalidArgument(
                                "'epsilon' in Op(LayerNorm) should be between"
                                "0.0 and 0.001, But received [%s].",
                                epsilon));
        });

    AddAttr<float>("dropout_rate", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(drop_p >= 0.0f && drop_p <= 1.0f,
                            true,
                            platform::errors::InvalidArgument(
                                "'dropout_rate' must be between 0.0 and 1.0."));
        });

    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<std::string>(
        "dropout_implementation",
        "[\"downgrade_in_infer\"|\"upscale_in_train\"]"
        "The meaning is the same as 'attn_dropout_implementation'.")
        .SetDefault("downgrade_in_infer")
        .AddCustomChecker([](const std::string &type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train",
              true,
              platform::errors::InvalidArgument(
                  "dropout_implementation can only be downgrade_in_infer or "
                  "upscale_in_train"));
        });
    AddAttr<std::string>("act_method", "act_method")
        .SetDefault("gelu")
        .AddCustomChecker([](const std::string &act_type) {
          PADDLE_ENFORCE_EQ(
              act_type == "gelu" || act_type == "geglu" || act_type == "relu" || act_type == "none",
              true,
              platform::errors::InvalidArgument(
                  "Only support `gelu`, `geglu`, `relu`, `none` activation in "
                  "FusedMultiTransformer. "));
        });

    AddAttr<bool>(
        "trans_qkvw",
        "Whether the weights of qkv should be transposed. If true,"
        "the shape eights of qkv should be [3, num_head, dim_head, dim_embed]."
        "Otherwise the shape of weights of qkv should be"
        "[dim_embed, 3, num_head, dim_head]")
        .SetDefault(true);

    AddAttr<int>(
        "ring_id",
        "ring id for tensor model parallel. distributed training and inference")
        .SetDefault(-1);
    // for moe layer
    AddAttr<int>(
        "topk",
        "gate's topk im moe")
        .SetDefault(2);
    AddAttr<int>(
        "mp_size",
        "mp size")
        .SetDefault(1);
    AddAttr<int>(
        "mp_rank",
        "mp rank")
        .SetDefault(0);
    AddAttr<int>(
        "num_expert",
        "experts num im moe")
        .SetDefault(1);
    AddAttr<int>(
        "world_size",
        "world size")
        .SetDefault(1);
    AddAttr<int>(
        "moe_ring_id",
        "experts communicate group's ring id")
        .SetDefault(1);
    AddAttr<bool>(
        "approximate",
        "approximate in expert compute gelu")
        .SetDefault(true);
    AddComment(R"DOC(fused multi transformer layers op)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_multi_transformer_moe,
    ops::FusedMultiTransformerMoeOp,
    ops::FusedMultiTransformerMoeOpOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

