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

#include <fstream>
#include <sstream>

#include "trie.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace framework {
#define ENFORCE PADDLE_ENFORCE

// Trie
int Trie::load(const std::string& dir, const uint32_t thr_num) {
    std::string list_file = dir + "/file_list";
    std::ifstream ifs(list_file.c_str());
    if (!ifs.is_open()) {
        printf("open file %s failed\n", list_file.c_str());
        return -1;
    }

    std::vector<File> files;
    std::string line;
    uint32_t node_num = 0;
    while (getline(ifs, line)) {
        std::stringstream ss(line);
        File file;
        ss >> file.filename;
        ss >> file.node_num;

        file.filename = dir + "/" + file.filename; 
        file.node_off = node_num;
        node_num += file.node_num;
        files.emplace_back(std::move(file));
    }
    printf("total file_num: %zu, node_num: %u\n", files.size(), node_num);

    {
        ScopedNanoTimer t("Trie stat");
        parallel_run_range(files.size(), thr_num,
                [this, &files](uint32_t thr_id, uint32_t start, uint32_t end) {
            for (uint32_t i = start; i < end; ++i) {
                stat_file(thr_id, files.at(i));
            }
        });
    }

    Node root;
    for (auto& file: files) {
        root.child.insert(root.child.end(), file.root.begin(), file.root.end());
    }

    {
        ScopedNanoTimer t("Trie resize");
        resize(node_num + 1); // +1 for root

        size_t off = root.child.size();
        for (size_t i = 0; i < files.size(); ++i) {
            mem_off(files[i].node_off + 1) = off; //+1 for root
            ENFORCE(files[i].node_num >= files[i].root.size());
            off += files[i].node_num - files[i].root.size();
        }
        ENFORCE(off == node_num);
    }

    {
        ScopedNanoTimer t("Trie load");
        parallel_run_range(files.size(), thr_num,
                [this, &files](uint32_t thr_id, uint32_t start, uint32_t end) {
            for (size_t i = start; i < end; ++i) {
                load_file(thr_id, files.at(i));
            }
        });
    }

    link(root);

    return 0;
}

void Trie::parse(std::string& line, Node& node, uint32_t off) {
    node.child.clear();

    char* str = const_cast<char*>(line.c_str());
    char* endstr = nullptr;
    size_t len = 0;

    node.id = std::strtoul(str, &endstr, 10) + off;
    str = endstr;
    ENFORCE(*str == '\t');
    ++str;

    node.label = std::strtoul(str, &endstr, 10);
    str = endstr;
    ENFORCE(*str == '\t');
    ++str;

    len = std::strtoul(str, &endstr, 10);
    str = endstr;
    for (size_t k = 0; k < len; ++k) {
        node.child.push_back(std::strtoul(str, &endstr, 10) + off);
        ENFORCE(str != endstr);
        str = endstr;
        ++str;
    }

    node.aleaf = std::strtoul(str, &endstr, 10);
    str = endstr;
    ENFORCE(*str == '\0');
}

void Trie::stat_file(uint32_t thr_id, File& file) {
    printf("stat file %s\n", file.filename.c_str());
    Node node;

    std::ifstream ifs(file.filename.c_str());
    ENFORCE(ifs.is_open(), "open file %s failed\n", file.filename.c_str());

    std::string line;
    getline(ifs, line);

    parse(line, node, file.node_off);
    file.root = std::move(node.child);
}

void Trie::load_file(uint32_t thr_id, File& file) {
    printf("load file %s\n", file.filename.c_str());

    std::ifstream ifs(file.filename.c_str());
    ENFORCE(ifs.is_open(), "open file %s failed\n", file.filename.c_str());

    Node node;
    std::string line;
    // don't link root
    if (getline(ifs, line)) {
        parse(line, node, file.node_off);
        file.root = std::move(node.child);
    }

    while(getline(ifs, line)) {
        parse(line, node, file.node_off);
        link(node);
    }
}

#undef ENFORCE
}  // end namespace framework
}  // end namespace paddle
