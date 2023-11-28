//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/trie_search_op.h"

namespace paddle {
namespace operators {

class TrieSearchStartOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->ShareDim("ids", "Out");
    ctx->ShareLoD("ids", "Out");
  }
};

class TrieSearchStartOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("parent_idx", "(Tensor), The parent_idx input tensor of trie_search_start op.");
    AddInput("ids", "(Tensor), The ids input tensor of trie_search_start op.");
    AddOutput("Out", "(Tensor), The output tensor of trie_search_start op.");
    AddComment(R"DOC(trie_search_start)DOC");
  }
};

class TrieSearchWaitOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
    ctx->HasInput("X"), true,
    platform::errors::NotFound(
        "Input(X) of TrieSearchWaitOp should not be null."));

    ctx->ShareDim("X", "Out");
    ctx->ShareLoD("X", "Out");
  }
};

class TrieSearchWaitOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of trie_search_wait op.");
    AddOutput("Out", "(Tensor), The output tensor of trie_search_wait op.");
    AddComment(R"DOC(trie_search_wait)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(trie_search_start, ops::TrieSearchStartOp,
                             ops::TrieSearchStartOpMaker)
REGISTER_OP_WITHOUT_GRADIENT(trie_search_wait, ops::TrieSearchWaitOp,
                             ops::TrieSearchWaitOpMaker)

REGISTER_OP_CPU_KERNEL(trie_search_start, ops::TrieSearchStartCPUKernel<int64_t>)
REGISTER_OP_CPU_KERNEL(trie_search_wait, ops::TrieSearchWaitCPUKernel<float>)