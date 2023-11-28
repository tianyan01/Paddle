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

#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <queue>

namespace paddle {
namespace framework {

template <class T>
void parallel_run_range(uint32_t n, uint32_t thr_num, T&& func) {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < thr_num; ++i) {
        threads.emplace_back(std::thread([i, n, thr_num, &func]() {
            func(i, n * i / thr_num, n * (i + 1) / thr_num);
        }));
    }
    for (auto& t : threads) {
        t.join();
    }
}

class ScopedNanoTimer {
public:
    ScopedNanoTimer(const std::string& n) : t0(std::chrono::high_resolution_clock::now()), m(n) {
    };
    ~ScopedNanoTimer() {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
        printf("%s cost %fs\n", m.c_str(), nanos/1000000000.0);
    }

protected:
    std::chrono::high_resolution_clock::time_point t0;
    std::string m;
};

class Trie {
struct File {
    std::string filename;
    std::vector<uint32_t> root;
    uint32_t node_num = 0;
    uint32_t node_off = 0;
};

struct Node {
    uint32_t id = 0;
    uint16_t label = 0;
    std::vector<uint32_t> child;
    uint8_t aleaf = 0;
};

public:
    Trie() {}
    virtual ~Trie() {}
    int load(const std::string& dir, const uint32_t thr_num=20u);

    uint16_t label(uint32_t id) {
        return label_.at(id);
    }

    uint8_t aleaf(uint32_t id) {
        return aleaf_.at(id);
    }

    void child(uint32_t id, std::vector<uint32_t>& child) {
        child.clear();
        size_t s = mem_off(id);
        size_t e = mem_off(id + 1);
        for (size_t i = s; i < e; ++i) {
            child.push_back(child_mem_.at(i));
        }
    }

    size_t child_size(uint32_t id) {
        size_t s = mem_off(id);
        size_t e = mem_off(id + 1);

        return e - s;
    }

    size_t child_at(uint32_t id, size_t i) {
        size_t s = mem_off(id);

        return child_mem_.at(s + i);
    }

    void print() {
        // level order traversal
        std::queue<uint32_t> q;
        q.push(0);
        std::vector<uint32_t> child;

        while(!q.empty()) {
            size_t len = q.size();
            for (size_t i = 0; i < len; ++i) {
                uint32_t id = q.front();
                q.pop();

                printf("[#%u,%u,%u,<", id, label(id), aleaf(id));
                this->child(id, child);
                for (auto j : child) {
                    q.push(j);
                    printf("#%u,", j);
                }
                printf(">] ");
            }
            printf("\n");
        }
    }

protected:
    void resize(uint32_t node_num) {
        label_.resize(node_num);
        aleaf_.resize(node_num);
        child_mem_.resize(node_num);
        mem_off_.resize(node_num + 1, 0);
    }

    uint32_t& mem_off(uint32_t id) {
        return mem_off_.at(id);
    }

    void link(const Node& node) {
        label_.at(node.id) = node.label;
        aleaf_.at(node.id) = node.aleaf;

        uint32_t addr = mem_off(node.id);
        for (size_t i = 0; i < node.child.size(); ++i) {
            child_mem_.at(addr++) = node.child[i];
        }
        if (mem_off(node.id + 1) == 0) {
            mem_off(node.id + 1) = addr;
        }
    }

    void parse(std::string& line, Node& node, uint32_t off=0);
    void load_file(uint32_t thr_id, File& file);
    void stat_file(uint32_t thr_id, File& file);

    std::vector<uint16_t> label_;
    std::vector<uint8_t>  aleaf_;
    std::vector<uint32_t> child_mem_;
    std::vector<uint32_t> mem_off_;
};

}  // end namespace framework
}  // end namespace paddle