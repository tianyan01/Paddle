/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

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

#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/dynload/cusparseLt.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace dyl = paddle::platform::dynload;
namespace paddle {
namespace operators {

#define CHECK_CUSPARSE(func)                                                   \
  do {                                                                         \
    cusparseStatus_t status = (func);                                          \
    CHECK(status == CUSPARSE_STATUS_SUCCESS)                                   \
        << "CUSPARSE API failed " << __FILE__ ":" << __LINE__ << "with error:" \
        << " " << status << std::endl;                                         \
  } while (0)

constexpr int alignment = 16;
class CompressWeight {
 public:
  CompressWeight(std::string w_name,
                 int w_m,
                 int w_n,
                 std::shared_ptr<memory::Allocation> weight) {
    name = w_name;
    m = w_m;
    n = w_n;
    data = weight;
  }
  std::string name;
  std::shared_ptr<memory::Allocation> data = nullptr;
  int m, n;
  ~CompressWeight() {}
};

// store compressed w
class WeightCache {
 public:
  static WeightCache& Instance() {
    static WeightCache instance;
    return instance;
  }
  ~WeightCache() { map_.clear(); }
  std::unordered_map<std::string, std::shared_ptr<CompressWeight>> map_;
  phi::Stream stream_;
  Place place_;
  bool init_ = false;
  // std::mutex cache_mutex_;
  std::shared_ptr<CompressWeight> find_weight(std::string w_name) {
    auto it = map_.find(w_name);
    if (it != map_.end()) {
      VLOG(3) << "CublasLtAlgoSelect Found in cache";
      return it->second;
    } else {
      return nullptr;
    }
  }

  void init(Place place, phi::Stream stream) {
    place_ = place;
    stream_ = stream;
    init_ = true;
  }

  std::shared_ptr<CompressWeight> create_compress_weight(
      cusparseLtHandle_t* handle,
      std::string w_name,
      int m,
      int n,
      cudaDataType_t type,
      void* src) {
    size_t compressed_size = 0;
    size_t buffer_size = 0;
    cusparseLtMatDescriptor_t mat;
    CHECK_CUSPARSE(dyl::cusparseLtStructuredDescriptorInit(
        handle,
        &mat,
        m,
        n,
        n,
        alignment,
        type,
        CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT));
    CHECK_CUSPARSE(dyl::cusparseLtSpMMACompressedSize2(
        handle,
        &mat,
        &compressed_size,
        &buffer_size));  // version 0.2 and version 0.4 is different
    // CHECK_CUSPARSE(dyl::cusparseLtSpMMACompressedSize2(
    //        handle, &mat, &compressed_size));// version 0.2 and version 0.4 is
    //        different
    auto compress_weight =
        memory::AllocShared(place_, compressed_size, stream_);
    auto temp = memory::AllocShared(place_, buffer_size, stream_);
    int* d_valid = reinterpret_cast<int*>(compress_weight->ptr());
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamSynchronize(reinterpret_cast<gpuStream_t>(stream_.id())));
    CHECK_CUSPARSE(
        dyl::cusparseLtSpMMAPruneCheck2(handle,
                                        &mat,
                                        0,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        src,
                                        d_valid,
                                        nullptr));
    int is_valid;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpy(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost));
    PADDLE_ENFORCE_EQ(
        is_valid,
        0,
        platform::errors::Fatal("weight  should be 2:4 column cutting "));
    int type_size = 4;  // for float
    if (type == CUDA_R_16F) {
      type_size = 2;
    }
    VLOG(0) << "before compress size  is " << (m * n * type_size)
            << " new size is " << compressed_size;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamSynchronize(reinterpret_cast<gpuStream_t>(stream_.id())));
    CHECK_CUSPARSE(dyl::cusparseLtSpMMACompress2(
        handle,
        &mat,
        0,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        src,
        compress_weight->ptr(),
        temp->ptr(),
        nullptr));  // version 0.2 and version 0.4 is different
    map_[w_name] = std::shared_ptr<CompressWeight>(
        new CompressWeight(w_name, m, n, compress_weight));
    CHECK_CUSPARSE(dyl::cusparseLtMatDescriptorDestroy(&mat));
    return map_[w_name];
  }

  std::shared_ptr<CompressWeight> create_compress_weight_int8(
      phi::DenseTensor x,
      cusparseLtHandle_t* handle,
      std::string w_name,
      int m,
      int n) {
    size_t compressed_size = 0;
    size_t buffer_size = 0;
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;

    CHECK_CUSPARSE(dyl::cusparseLtStructuredDescriptorInit(
        handle,
        &matB,
        m,
        n,
        n,
        alignment,
        CUDA_R_8I,
        CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT));
    CHECK_CUSPARSE(dyl::cusparseLtDenseDescriptorInit(
        handle, &matA, n, n, n, alignment, CUDA_R_8I, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(dyl::cusparseLtDenseDescriptorInit(
        handle, &matC, n, m, m, alignment, CUDA_R_16F, CUSPARSE_ORDER_ROW));
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE(
        dyl::cusparseLtMatmulDescriptorInit(handle,
                                            &matmul,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_TRANSPOSE,
                                            &matA,
                                            &matB,
                                            &matC,
                                            &matC,
                                            CUSPARSE_COMPUTE_32I));
    CHECK_CUSPARSE(dyl::cusparseLtMatmulAlgSelectionInit(
        handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
    CHECK_CUSPARSE(
        dyl::cusparseLtMatmulPlanInit(handle, &plan, &matmul, &alg_sel));

    CHECK_CUSPARSE(dyl::cusparseLtSpMMACompressedSize(
        handle, &plan, &compressed_size, &buffer_size));
    auto compress_weight =
        memory::AllocShared(place_, compressed_size, stream_);
    auto temp = memory::AllocShared(place_, buffer_size, stream_);
    int* d_valid = reinterpret_cast<int*>(compress_weight->ptr());
    void* src = x.data<int8_t>();
    CHECK_CUSPARSE(dyl::cusparseLtSpMMAPruneCheck2(
        handle, &matB, 0, CUSPARSE_OPERATION_TRANSPOSE, src, d_valid, nullptr));
    int is_valid;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpy(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost));
    PADDLE_ENFORCE_EQ(
        is_valid,
        0,
        platform::errors::Fatal("weight  should be 2:4 column cutting "));
    VLOG(0) << "before compress size  is " << (m * n) * sizeof(int8_t)
            << " new size is " << compressed_size;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamSynchronize(reinterpret_cast<gpuStream_t>(stream_.id())));
    CHECK_CUSPARSE(dyl::cusparseLtSpMMACompress(
        handle, &plan, src, compress_weight->ptr(), temp->ptr(), nullptr));
    map_[w_name] = std::shared_ptr<CompressWeight>(
        new CompressWeight(w_name, m, n, compress_weight));
    CHECK_CUSPARSE(dyl::cusparseLtMatDescriptorDestroy(&matA));
    CHECK_CUSPARSE(dyl::cusparseLtMatDescriptorDestroy(&matB));
    CHECK_CUSPARSE(dyl::cusparseLtMatDescriptorDestroy(&matC));
    CHECK_CUSPARSE(dyl::cusparseLtMatmulPlanDestroy(&plan));
    return map_[w_name];
  }
};

// to record the fast algo,split_k(add in version 0.3)  for A[m*n] * B[n*k]  (B
// is 2:4)
struct CuSparseLtAlgoParam {
  int m;
  int n;
  int k;
  int type;
  int id;              // CUSPARSELT_MATMUL_ALG_CONFIG_ID
  int32_t is_split_k;  // CUSPARSELT_MATMUL_SPLIT_K
  cusparseLtSplitKMode_t
      split_k_mode;        // CUSPARSELT_MATMUL_SPLIT_K_MODE value is
                           // CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL or
                           // CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS
  int32_t split_k_buffer;  // CUSPARSELT_MATMUL_SPLIT_K_BUFFERS
};

class CuSparseLtAlgoCache {
 public:
  static CuSparseLtAlgoCache& Instance() {
    static CuSparseLtAlgoCache instance;
    return instance;
  }

  CuSparseLtAlgoCache() {
    // Init map_ from cache file
    std::ifstream infile;
    infile.open(config_filename_);
    if (!infile.is_open()) {
      VLOG(0) << "No CuSparseLtAlgoCache file found";
      return;
    }
    int64_t seed = 0;
    struct CuSparseLtAlgoParam algo_param;
    int split_k_mode;
    while (!infile.eof()) {
      infile >> seed >> algo_param.m >> algo_param.n >> algo_param.k >>
          algo_param.type >> algo_param.id >> algo_param.is_split_k >>
          split_k_mode >> algo_param.split_k_buffer;
      map_[seed] = algo_param;
    }
    algo_param.split_k_mode = (cusparseLtSplitKMode_t)(split_k_mode);
    VLOG(0) << " sucess load CuSparseLtAlgoCache file, num is " << map_.size();
    infile.close();
  }

  int64_t getSeed(int m, int n, int k, cudaDataType_t type) {
    int64_t seed = 0;
    std::hash<int64_t> hash_fn;
    int type_num = 2;
    if (type == CUDA_R_8I) {
      type_num = 0;
    } else if (type == CUDA_R_16F) {
      type_num = 1;
    }

    HashValue_(&seed, hash_fn, (static_cast<int64_t>(m)));
    HashValue_(&seed, hash_fn, (static_cast<int64_t>(n)));
    HashValue_(&seed, hash_fn, (static_cast<int64_t>(k)));
    HashValue_(&seed, hash_fn, (static_cast<int64_t>(type_num)));
    return seed;
  }

  // search A[m*n] * B[n*k] AlgoId and A B type is type and A,B is not
  // transposition
  CuSparseLtAlgoParam* getAlgoParam(
      cusparseLtHandle_t* handle, int m, int n, int k, cudaDataType_t type) {
    int64_t seed = getSeed(m, n, k, type);
    auto it = map_.find(seed);
    if (it != map_.end()) {
      return &(it->second);
    }
    return nullptr;
  }
  // (m*k) * (k*n) = (m*n)
  CuSparseLtAlgoParam* CuSparseLtAlgoSelect(
      cusparseLtHandle_t* handle,
      cusparseLtMatmulPlan_t* plan,
      cusparseLtMatmulAlgSelection_t* algSelection,
      int m,
      int n,
      int k,
      cudaDataType_t type,
      const void* alpha,
      const void* dA,
      const void* dB,
      const void* beta,
      const void* dC,
      void* dD,
      cudaStream_t* streams,
      int32_t numStreams = 1) {
    auto algo_param_ptr = getAlgoParam(handle, m, n, k, type);
    if (algo_param_ptr != nullptr) {
      return algo_param_ptr;
    }
    // clear kernel on gpu to avoid interfering with algorithm search
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(streams[0]));

    // int split_k = 1;
    // CHECK_CUSPARSE(dyl::cusparseLtMatmulAlgSetAttribute(
    //                   handle, algSelection, CUSPARSELT_MATMUL_SPLIT_K,
    //                   &split_k, sizeof(split_k)));

    VLOG(0) << "start to cusparseLtMatmulSearch for " << m << " " << n << " "
            << k;
    CHECK_CUSPARSE(dyl::cusparseLtMatmulSearch(handle,
                                               plan,
                                               alpha,
                                               dA,
                                               dB,
                                               beta,
                                               dC,
                                               dD,
                                               nullptr,
                                               streams,
                                               numStreams));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(streams[0]));
    struct CuSparseLtAlgoParam algo_param;
    algo_param.m = m;
    algo_param.n = n;
    algo_param.k = k;
    int type_num = 2;
    if (type == CUDA_R_8I) {
      type_num = 0;
    } else if (type == CUDA_R_16F) {
      type_num = 1;
    }
    algo_param.type = type_num;
    CHECK_CUSPARSE(
        dyl::cusparseLtMatmulAlgGetAttribute(handle,
                                             algSelection,
                                             CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                             &algo_param.id,
                                             sizeof(algo_param.id)));
    CHECK_CUSPARSE(
        dyl::cusparseLtMatmulAlgGetAttribute(handle,
                                             algSelection,
                                             CUSPARSELT_MATMUL_SPLIT_K,
                                             &algo_param.is_split_k,
                                             sizeof(algo_param.is_split_k)));
    CHECK_CUSPARSE(
        dyl::cusparseLtMatmulAlgGetAttribute(handle,
                                             algSelection,
                                             CUSPARSELT_MATMUL_SPLIT_K_MODE,
                                             &algo_param.split_k_mode,
                                             sizeof(algo_param.split_k_mode)));
    CHECK_CUSPARSE(dyl::cusparseLtMatmulAlgGetAttribute(
        handle,
        algSelection,
        CUSPARSELT_MATMUL_SPLIT_K_BUFFERS,
        &algo_param.split_k_buffer,
        sizeof(algo_param.split_k_buffer)));
    auto seed = getSeed(m, n, k, type);
    map_[seed] = algo_param;
    VLOG(2) << " for " << m << "," << n << "," << k << " "
            << "id " << algo_param.id << " is_split_k is "
            << algo_param.is_split_k << " split_k_mode "
            << static_cast<int>(algo_param.split_k_mode)
            << " split_k_buffer is " << algo_param.split_k_buffer;

    return &map_[seed];
  }

  ~CuSparseLtAlgoCache() {
    // Serialize map_ to cache file
    std::ofstream outfile;
    outfile.open(config_filename_, std::ios::out | std::ios::trunc);
    for (const auto p : map_) {
      outfile << p.first << " ";
      outfile << p.second.m << " ";
      outfile << p.second.n << " ";
      outfile << p.second.k << " ";
      outfile << int(p.second.type) << " ";
      outfile << p.second.id << " ";
      outfile << p.second.is_split_k << " ";
      outfile << int(p.second.split_k_mode) << " ";
      outfile << p.second.split_k_buffer << " ";
      outfile << std::endl;
    }
    outfile.close();
  }

 private:
  void HashValue_(int64_t* seed,
                  const std::hash<int64_t>& hash_fn,
                  int64_t value) {
    *seed ^= hash_fn(value) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  }
  std::string config_filename_{"./paddle_cusparselt_cache"};
  std::unordered_map<int64_t, CuSparseLtAlgoParam> map_;
  std::mutex cache_mutex_;
};

}  // namespace operators
}  // namespace paddle
