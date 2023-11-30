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

#include "paddle/fluid/framework/trie_manager.h"
#include <valarray>

namespace paddle {
namespace framework {
std::shared_ptr<TrieManager> TrieManager::_s_instance = nullptr;

void TrieManager::reset() {
    VLOG(3) << "trie reset...";
    std::unique_lock<std::mutex> lock(mtx_);

    // Todo: cache level1
    parent_idx_.mutable_data<int64_t>({1}, phi::GPUPinnedPlace());
    int64_t* parent_idx = parent_idx_.data<int64_t>();
    parent_idx[0] = 0;

    select_ids_.mutable_data<int64_t>({1}, phi::GPUPinnedPlace());
    int64_t* select_ids = select_ids_.data<int64_t>();
    select_ids[0] = 0;

    label2node_.resize(1);
    label2node_[0][0] = 0;

    phase_ = Phase::run;
    cv_.notify_one();
}

void TrieManager::search_start(const Tensor* d_parent, const Tensor* d_select) {
    VLOG(3) << "trie search_start: " << d_parent->dims() <<  " # " << d_select->dims();
    if (d_parent->numel() == 0) {
        return;
    }

    std::unique_lock<std::mutex> lock(mtx_);

    TensorCopySync(*d_parent, phi::GPUPinnedPlace(), &parent_idx_);
    TensorCopySync(*d_select, phi::GPUPinnedPlace(), &select_ids_);

    phase_ = Phase::run;
    cv_.notify_one();
}

void TrieManager::search_wait() {
    VLOG(3) << "trie search_wait";

    std::unique_lock<std::mutex> lock(mtx_);
    while (phase_ !=  Phase::done) {
        cv_.wait(lock);
    }
}

void TrieManager::run() {
    uint32_t thr_num = 10;

    for (;;) {
        std::unique_lock<std::mutex> lock(mtx_);
        while (phase_ !=  Phase::run) {
            cv_.wait(lock);
            if (phase_ == Phase::stop) {
                VLOG(3) << "phase_ == stop and exit";
                return;
            }
        }
        VLOG(3) << "trie run, parent_idx: " << parent_idx_ << "\n, select_ids: " << select_ids_;
        VLOG(3) << "label2node_ size: " << label2node_.size();

        // 1.
        int numel = parent_idx_.numel();
        PADDLE_ENFORCE(numel == select_ids_.numel());
        int64_t* parent_idx = parent_idx_.data<int64_t>();
        int64_t* select_ids = select_ids_.data<int64_t>();

        std::vector<std::unordered_map<uint16_t, uint32_t>> label2node(numel);
        std::vector<std::vector<uint16_t>> outs(numel);
        parallel_run_range(numel, thr_num, [this, parent_idx, select_ids, &outs, &label2node] (
                uint32_t thr_id, uint32_t start, uint32_t end) {
            for (size_t i = start; i < end; ++i) {
                auto& out = outs.at(i);

                int64_t idx = parent_idx[i];
                if (label2node_.size() == 1) {
                    idx = 0;
                }

                auto& l2n_ = label2node_.at(idx);
                auto& l2n = label2node.at(i);
                auto it = l2n_.find(select_ids[i]);
                if (it == l2n_.end()) {
                    out.push_back(endid_);
                    l2n.insert({endid_, end_nodeid_});
                    continue;
                }
                if (it->second == end_nodeid_) {
                    out.push_back(endid_);
                    l2n.insert({endid_, end_nodeid_});
                    continue;
                }

                size_t chs = trie_.child_size(it->second);
                if (chs == 0 || trie_.aleaf(it->second)) {
                    out.push_back(endid_);
                    l2n.insert({endid_, end_nodeid_});
                }

                for (size_t j = 0; j < chs; ++j) {
                    uint32_t cid = trie_.child_at(it->second, j);
                    uint32_t lab = trie_.label(cid);

                    out.push_back(lab);
                    l2n.insert({lab, cid});
                }
            }
        });
        label2node_.swap(label2node);

        numel = 0;
        for (size_t i = 0; i < outs.size(); ++i) {
            numel += outs[i].size();
        }

        auto next_out = next_out_.mutable_data<int64_t>({numel, 1}, phi::GPUPinnedPlace());
        auto next_lod = next_lod_.mutable_data<int64_t>({int(outs.size())+1, 1}, phi::GPUPinnedPlace());

        // 2.
        next_lod[0] = 0;
        int k = 0;
        for (size_t i = 0; i < outs.size(); ++i) {
            for (size_t j = 0; j < outs[i].size(); ++j) {
                next_out[k] = outs[i][j];
                ++k;
            }
            next_lod[i+1] = next_lod[i] + int64_t(outs[i].size());
        }

        VLOG(10) << "out " << next_out_ << "\n lod " << next_lod_;
        VLOG(3) << "out" << framework::PrintTensor<int64_t>(next_out_, 100);

        // 3.
        TensorCopySync(next_out_, place_, &next_out_d_);
        TensorCopySync(next_lod_, place_, &next_lod_d_);

        phase_ = Phase::done;
        cv_.notify_one();
    }
}

}  // end namespace framework
}  // end namespace paddle
