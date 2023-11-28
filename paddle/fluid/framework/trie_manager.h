/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "trie.h"

namespace paddle {
namespace framework {

template <typename T>
std::string PrintTensor(const framework::Tensor& tensor,
                        int64_t print_len = 2000) {
    std::stringstream sstream;

    if (print_len == -1) {
        print_len = tensor.numel();
    }
    print_len = std::min(tensor.numel(), print_len);

    const T* data = nullptr;
    if (platform::is_cpu_place(tensor.place())) {
        data = tensor.data<T>();
    } else {
        framework::Tensor cpu_tensor;
        platform::CPUPlace cpu_place;
        TensorCopySync(tensor, cpu_place, &cpu_tensor);
        data = cpu_tensor.data<T>();
    }

    sstream << "\n  - dtype: " << tensor.dtype();
    sstream << "\n  - place: " << tensor.place();
    sstream << "\n  - dims: " << tensor.dims();
    sstream << "\n  - data: [";
    if (print_len > 0) {
        sstream << data[0];
        for (int64_t i = 1; i < print_len; ++i) {
            sstream << " " << data[i];
        }
    }
    sstream << "]" << std::endl;

    return sstream.str();
}

class TrieManager {
enum class Phase {
    init,
    run,
    done,
    stop
};

public:
    TrieManager(uint16_t endid) : endid_(endid),
            place_(platform::GetCurrentDeviceId()) {
        thread_ = std::thread(&TrieManager::run, this);
    }

    ~TrieManager() {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            phase_ = Phase::stop;
        }
        cv_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    static std::shared_ptr<TrieManager> GetInstance() {
        PADDLE_ENFORCE_EQ(
            _s_instance == nullptr,
            false,
            platform::errors::PreconditionNotMet(
                "GetInstance failed in TrieManager, you should use SetInstance firstly"));
        return _s_instance;
    }

    static std::shared_ptr<TrieManager> SetInstance(uint16_t endid) {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (nullptr == _s_instance) {
            VLOG(3) << "TrieManager _s_instance is null";
            _s_instance.reset(new TrieManager(endid));
        } else {
            LOG(WARNING) << "You have already used TrieManager SetInstance() before";
        }

        return _s_instance;
    }

    int load(const std::string& dir, const uint32_t thr_num=20) {
        return trie_.load(dir, thr_num);
    }
    void reset();
    void search_start(const Tensor* d_parent, const Tensor* d_select);
    void search_wait();

    // gpu
    Tensor next_out_d_;
    Tensor next_lod_d_;

protected:
    static std::shared_ptr<TrieManager> _s_instance;

    // cpu
    Tensor parent_idx_;
    Tensor select_ids_;
    std::vector<std::unordered_map<uint16_t, uint32_t>> label2node_;

    // cpu
    Tensor next_out_;
    Tensor next_lod_;

    Trie trie_;
    size_t endid_;
    size_t vocab_size_;

    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread thread_;
    Phase phase_ = Phase::init;

    platform::CUDAPlace place_;

    void run();
};

}  // end namespace framework
}  // end namespace paddle