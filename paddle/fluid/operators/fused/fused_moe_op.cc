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

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class FusedMoeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    // input
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "fused_moe");
    OP_INOUT_CHECK(context->HasInput("GateWeight"),
                   "Input",
                   "GateWeight",
                   "fused_moe");
    OP_INOUT_CHECK(context->HasInput("GateBias"),
                   "Input",
                   "GateBias",
                   "fused_moe");
    OP_INOUT_CHECK(context->HasInput("LnScale"),
                   "Input",
                   "LnScale",
                   "fused_moe");
    OP_INOUT_CHECK(context->HasInput("LnBias"),
                   "Input",
                   "LnBias",
                   "fused_moe");
    OP_INOUT_CHECK(context->HasInputs("ExpertsWeight1"),
                   "Input",
                   "ExpertsWeight1",
                   "fused_moe");
    OP_INOUT_CHECK(context->HasInputs("ExpertsBias1"),
                   "Input",
                   "ExpertsBias1",
                   "fused_moe");
    OP_INOUT_CHECK(context->HasInputs("ExpertsWeight2"),
                   "Input",
                   "ExpertsWeight2",
                   "fused_moe");
    OP_INOUT_CHECK(context->HasInputs("ExpertsBias2"),
                   "Input",
                   "ExpertsBias2",
                   "fused_moe");
    // output
    OP_INOUT_CHECK(context->HasOutput("Out"),
                   "Output",
                   "Out",
                   "fused_moe");
    auto x_dims = context->GetInputDim("X");
    context->SetOutputDim("Out", x_dims);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class FusedMoeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // AsDispensable 可有可无
    // AsDuplicable 可复制
    // input
    AddInput("X", "The input of FusedMoe op");
    AddInput("GateWeight", "The gate weight of FusedMoe op");
    AddInput("GateBias", "The gate bias of FusedMoe op");
    AddInput("LnScale", "The ln scale of FusedMoe op");
    AddInput("LnBias", "The LnBias of FusedMoe op");
    AddInput("ExpertsWeight1", "The expert linear1 weights of fused_moe op")
        .AsDuplicable();
    AddInput("ExpertsBias1", "The expert linear1 biases of fused_moe op")
        .AsDuplicable()
        .AsDispensable();
    AddInput("ExpertsWeight2", "The expert linear2 weights of fused_moe op")
        .AsDuplicable();
    AddInput("ExpertsBias2", "The expert linear2 biases of fused_moe op")
        .AsDuplicable()
        .AsDispensable();
    // output
    AddOutput("Out", "Out");
    // attr
    AddAttr<bool>("pre_layer_norm", "pre_layer_norm").SetDefault(true);
    AddAttr<float>("ln_epsilon", "ln_epsilon").SetDefault(1e-5f);
    AddAttr<int>("topk", "top k in gate").SetDefault(2);
    AddAttr<int>("mp_size", "mp_size").SetDefault(1);
    AddAttr<int>("mp_rank", "mp_rank").SetDefault(0);
    AddAttr<int>("num_expert", "num_expert").SetDefault(1);
    AddAttr<int>("world_size", "world_size").SetDefault(1);
    AddAttr<int>("moe_ring_id", "moe_ring_id").SetDefault(-1);
    AddComment(R"DOC(
      The fused_moe operator is the same as the following pseudo codes:
      
      pass
      
      )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_moe,
                  ops::FusedMoeOp,
                  ops::FusedMoeOpMaker,
                  paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
                  paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);