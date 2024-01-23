/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/fusion/cutlass/utils/cuda_utils.h"

namespace phi {

enum class ActivationType {
    Gelu,
    Relu,
    Silu,
    GeGLU,
    ReGLU,
    SiGLU,
    Identity,
    InvalidType
};

inline ActivationType getActivationType(const std::string &activation_type_str)
{
    if (activation_type_str == "Gelu" || activation_type_str == "gelu") {
        return ActivationType::Gelu;
    } else if (activation_type_str == "Relu" || activation_type_str == "relu") {
        return ActivationType::Relu;
    } else if (activation_type_str == "None" || activation_type_str == "none") {
    	return ActivationType::Identity;
    }
    else {
    	PADDLE_THROW(phi::errors::Unimplemented(
    			"Activation Type: " + activation_type_str + " not supported !"));
    }
    return ActivationType::InvalidType;
}

inline bool isGatedActivation(ActivationType activaiton_type)
{
    return activaiton_type == ActivationType::GeGLU || activaiton_type == ActivationType::ReGLU
           || activaiton_type == ActivationType::SiGLU;
}

}  // namespace fastertransformer
