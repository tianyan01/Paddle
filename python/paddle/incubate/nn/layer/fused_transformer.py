# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from paddle.nn import functional as F
from paddle.incubate.nn import functional as incubate_f
from paddle.nn import Layer
from paddle.framework import ParamAttr
import paddle
import paddle.nn as nn
from paddle import _legacy_C_ops, _C_ops
from paddle.nn import ParameterList
from paddle.nn.layer.transformer import (
    _convert_attention_mask,
    _convert_param_attr_to_list,
)
from paddle.nn.initializer import Constant
from paddle.fluid.dygraph import no_grad
from paddle.fluid.framework import convert_np_dtype_to_dtype_, _non_static_mode
from paddle.fluid.core import VarDesc
from paddle.fluid import core
import numpy as np


# for distributed tensor model parallel
def _set_var_distributed(var):
    if var is None:
        return

    var.is_distributed = True

    if not _non_static_mode():
        # NOTE: use current_block and find_var_recursive to support while_loop
        startup_block = paddle.static.default_startup_program().current_block()
        main_block = paddle.static.default_main_program().current_block()
        startup_block._find_var_recursive(var.name).is_distributed = True
        main_block._find_var_recursive(var.name).is_distributed = True


def _to_dtype(t, dtype):
    # this function is a prune of Layer._transform function to fix fused op under amp.decorator(O2)
    if dtype == t.dtype:
        return t

    if type(dtype) is not VarDesc.VarType:
        dtype = convert_np_dtype_to_dtype_(dtype)

    if t.place.is_gpu_place():
        size_dtype = core.size_of_dtype(dtype)
        waiting_alloc_memory = (
            ((np.prod(t.shape) * size_dtype) / 256 + 1) * 256 * 1.2
        )
        gpu_memory_available = core.gpu_memory_available()
        if gpu_memory_available < waiting_alloc_memory:
            t_used = t._copy_to(paddle.CPUPlace(), False)
            t.value().get_tensor()._clear()
        else:
            t_used = t
    else:
        t_used = t

    if dtype is not None and dtype != t_used.dtype:
        with paddle.fluid.framework._dygraph_place_guard(place=t_used.place):
            t_casted = t_used.cast(dtype=dtype)
    else:
        t_casted = t_used

    new_t = t_casted

    dst_tensor = t.value().get_tensor()
    src_tensor = new_t.value().get_tensor()
    dst_tensor._share_data_with(src_tensor)

    return t


class FusedBiasDropoutResidualLayerNorm(Layer):
    """
    Applies fused_bias_dropout_residual_layer_norm operation.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout after attention.
            0 for no dropout. Default 0.5.
        bias_attr (ParamAttr|bool, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            If it is set to False, this layer will not have trainable bias parameter.
            See usage for details in :code:`ParamAttr`.
        epsilon (float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.

    Examples:

        .. code-block:: python

            # required: gpu
            import paddle
            # input: [batch_size, seq_len, embed_dim]
            x = paddle.rand((2, 4, 128))
            # residual: [batch_size, seq_len, embed_dim]
            residual = paddle.rand((2, 4, 128))
            fused_bias_dropout_residual_ln = paddle.incubate.nn.FusedBiasDropoutResidualLayerNorm(128)
            output = fused_bias_dropout_residual_ln(x, residual)  # [2, 4, 128]
    """

    def __init__(
        self,
        embed_dim,
        dropout_rate=0.5,
        weight_attr=None,
        bias_attr=None,
        epsilon=1e-5,
        name=None,
    ):
        super(FusedBiasDropoutResidualLayerNorm, self).__init__()
        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            "but recieved {}".format(embed_dim)
        )
        self._dtype = self._helper.get_default_dtype()
        self._bias_attr = bias_attr
        self._weight_attr = weight_attr
        self.embed_dim = embed_dim
        self.linear_bias = self.create_parameter(
            shape=[embed_dim],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )
        self.ln_scale = self.create_parameter(
            attr=self._weight_attr,
            shape=[embed_dim],
            default_initializer=Constant(value=1.0),
        )
        self.ln_bias = self.create_parameter(
            attr=self._bias_attr, shape=[embed_dim], is_bias=True
        )
        self.dropout_rate = dropout_rate
        self._epsilon = epsilon

        self.name = name

    def forward(self, x, residual):
        """
        Applies fused_bias_dropout_residual_layer_norm operation.

        Parameters:
            x (Tensor): The input tensor. It is a tensor with shape
                `[batch_size, seq_len, embed_dim]`. The data type should be
                float32 or float64.
            residual (Tensor, optional): The residual tensor. It is a tensor
                with shape `[batch_size, value_length, vdim]`. The data type
                should be float32 or float64.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `x`.
        """

        out = incubate_f.fused_bias_dropout_residual_layer_norm(
            x=x,
            residual=residual,
            bias=self.linear_bias,
            ln_scale=self.ln_scale,
            ln_bias=self.ln_bias,
            dropout_rate=self.dropout_rate,
            ln_epsilon=self._epsilon,
            training=self.training,
            mode='upscale_in_train',
            name=self.name,
        )
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'embed_dim={}, seq_len={}, dropout_rate={}, epsilon={}, dtype={}{}'.format(
            self.embed_dim,
            self.seq_len,
            self.dropout_rate,
            self._epsilon,
            self._dtype,
            name_str,
        )


class FusedMultiHeadAttention(Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.
    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout after attention.
            0 for no dropout. Default 0.5.
        attn_dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout in attention.
            0 for no dropout. Default 0.5.
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        normalize_before (bool, optional): Indicate  whether it is pre_layer_norm
            (True) or post_layer_norm architecture (False). Default False.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Now, only False is supported. Default False.
        qkv_weight_attr(ParamAttr, optional): To specify the weight parameter property
            for QKV projection computation. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        qkv_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for QKV projection computation. The `False` value means the corresponding layer
            would not have trainable bias parameter. Default: None, which means the
            default bias parameter property is used. See usage for details in :code:`ParamAttr`.
        linear_weight_attr(ParamAttr, optional): To specify the weight parameter property
            for linear projection computation. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for linear projection computation. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        pre_ln_scale_attr(ParamAttr, optional): To specify the weight parameter property
            for pre_layer_norm computation. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        pre_ln_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for pre_layer_norm computation. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln_scale_attr(ParamAttr, optional): To specify the weight parameter property
            for post_layer_norm computation. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for post_layer_norm computation. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        epsilon (float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        nranks (int, optional): Distributed tensor model parallel nranks. Default is 1, means not using tensor parallel.
        ring_id (int, optional): For distributed tensor model parallel. Default is -1, means not using tensor parallel.

    Examples:

        .. code-block:: python

            # required: gpu
            import paddle
            # input: [batch_size, sequence_length, embed_dim]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.incubate.nn.FusedMultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout_rate=0.5,
        attn_dropout_rate=0.5,
        dropout_seed=None,
        attn_dropout_seed=None,
        kdim=None,
        vdim=None,
        normalize_before=False,
        need_weights=False,
        qkv_weight_attr=None,
        qkv_bias_attr=None,
        linear_weight_attr=None,
        linear_bias_attr=None,
        pre_ln_scale_attr=None,
        pre_ln_bias_attr=None,
        ln_scale_attr=None,
        ln_bias_attr=None,
        epsilon=1e-5,
        nranks=1,
        ring_id=-1,
        name=None,
    ):
        super(FusedMultiHeadAttention, self).__init__()

        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            "but received {}".format(embed_dim)
        )
        assert (
            num_heads > 0
        ), "Expected nhead to be greater than 0, " "but received {}".format(
            num_heads
        )

        self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        self._epsilon = epsilon
        self._ring_id = ring_id

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim
        self.vdim = vdim
        self.need_weights = need_weights
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        assert need_weights is False, "Only support need_weight is False now."

        # tensor model parallel
        assert num_heads % nranks == 0
        num_heads = num_heads // nranks

        self.qkv_weight = self.create_parameter(
            shape=[3, num_heads, self.head_dim, embed_dim],
            attr=qkv_weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        self.qkv_bias = self.create_parameter(
            shape=[3, num_heads, self.head_dim],
            attr=qkv_bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )
        self.linear_weight = self.create_parameter(
            shape=[num_heads * self.head_dim, embed_dim],
            attr=linear_weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        self.linear_bias = self.create_parameter(
            shape=[embed_dim],
            attr=linear_bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
            # column parallel
            _set_var_distributed(self.qkv_weight)
            _set_var_distributed(self.qkv_bias)
            # row parallel
            _set_var_distributed(self.linear_weight)

        if normalize_before:
            self.pre_ln_scale = self.create_parameter(
                attr=pre_ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
            )
            self.pre_ln_bias = self.create_parameter(
                attr=pre_ln_bias_attr, shape=[embed_dim], is_bias=True
            )
            self.ln_scale = None
            self.ln_bias = None
        else:
            self.pre_ln_scale = None
            self.pre_ln_bias = None
            self.ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
            )
            self.ln_bias = self.create_parameter(
                attr=ln_bias_attr, shape=[embed_dim], is_bias=True
            )

        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.dropout_seed = dropout_seed
        self.attn_dropout_seed = attn_dropout_seed

        self.name = name

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        """
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
            cache (MultiHeadAttention.Cache|MultiHeadAttention.StaticCache, optional):
                Now, only None is supported. Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output.
        """
        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, query.dtype)

        out = incubate_f.fused_multi_head_attention(
            x=query,
            qkv_weight=self.qkv_weight,
            linear_weight=self.linear_weight,
            pre_layer_norm=self.normalize_before,
            pre_ln_scale=self.pre_ln_scale,
            pre_ln_bias=self.pre_ln_bias,
            ln_scale=self.ln_scale,
            ln_bias=self.ln_bias,
            pre_ln_epsilon=self._epsilon,
            qkv_bias=self.qkv_bias,
            linear_bias=self.linear_bias,
            cache_kv=cache,
            attn_mask=attn_mask,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
            dropout_seed=self.dropout_seed,
            attn_dropout_seed=self.attn_dropout_seed,
            ln_epsilon=self._epsilon,
            training=self.training,
            ring_id=self._ring_id,
            name=self.name,
        )
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'embed_dim={}, num_heads={}, dropout_rate={}, attn_dropout_rate={}, epsilon={}, kdim={}, vdim={}, normalize_before={}, need_weights={}, dtype={}{}'.format(
            self.embed_dim,
            self.num_heads,
            self.dropout_rate,
            self.attn_dropout_rate,
            self._epsilon,
            self.kdim,
            self.vdim,
            self.normalize_before,
            self.need_weights,
            self._dtype,
            name_str,
        )

    def _amp_decorate(self, dtype):
        # tmp fix for amp.decorator(O2)
        layer_norm_params_id = []
        if self.normalize_before:
            layer_norm_params_id.append(id(self.pre_ln_scale))
            layer_norm_params_id.append(id(self.pre_ln_bias))
        else:
            layer_norm_params_id.append(id(self.ln_scale))
            layer_norm_params_id.append(id(self.ln_bias))

        for key, param in self._parameters.items():
            if id(param) in layer_norm_params_id:
                continue
            if param is not None:
                with no_grad():
                    param_applied = _to_dtype(param, dtype)

        self._dtype = dtype


class FusedFeedForward(Layer):
    """
    Parameters:
        d_model (int): The expected feature size in the input and output.
        dim_feedforward (int): The hidden layer size.
        dropout_rate (float, optional): The dropout probability used in pre-process
            and post-precess. Default 0.1
        epsilon (float, optional): he small value added to the variance to prevent
            division by zero. Default: 1e-05.
        activation (str, optional): The activation function. Default relu.
        act_dropout_rate (float, optional): The dropout probability after activition.
            If None, use the value of `dropout_rate`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into, preprocessing or postprocessing. Default False
        linear1_weight_attr(ParamAttr, optional): To specify the weight parameter property
            for FFN first linear. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear1_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for FFN first linear. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear2_weight_attr(ParamAttr, optional): To specify the weight parameter property
            for FFN second linear. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear2_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for FFN second linear. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln1_scale_attr(ParamAttr, optional): To specify the weight parameter property
            for FFN pre_layer_norm. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln1_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for FFN pre_layer_norm. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln2_scale_attr(ParamAttr, optional): To specify the weight parameter property
            for FFN post_layer_norm. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln2_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for FFN layer_norm. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        nranks (int, optional): Distributed tensor model parallel nranks. Default is 1, means not using tensor parallel.
        ring_id (int, optional): For distributed tensor model parallel. Default is -1, means not using tensor parallel.
        name (str, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn import FusedFeedForward

            fused_feedforward_layer = FusedFeedForward(8, 8)
            x = paddle.rand((1, 8, 8))
            out = fused_feedforward_layer(x)
            print(out.numpy().shape)
            # (1, 8, 8)
    """

    def __init__(
        self,
        d_model,
        dim_feedforward,
        dropout_rate=0.1,
        epsilon=1e-05,
        activation="relu",
        act_dropout_rate=None,
        seed=None,
        normalize_before=False,
        linear1_weight_attr=None,
        linear1_bias_attr=None,
        linear2_weight_attr=None,
        linear2_bias_attr=None,
        ln1_scale_attr=None,
        ln1_bias_attr=None,
        ln2_scale_attr=None,
        ln2_bias_attr=None,
        nranks=1,
        ring_id=-1,
        name=None,
    ):

        super(FusedFeedForward, self).__init__()
        assert (
            d_model > 0
        ), "Expected d_model to be greater than 0, but received {}".format(
            d_model
        )
        assert (
            dim_feedforward > 0
        ), "Expected dim_feedforward to be greater than 0, but received {}".format(
            dim_feedforward
        )

        self._dtype = self._helper.get_default_dtype()
        self._d_model = d_model

        assert dim_feedforward % nranks == 0
        dim_feedforward = dim_feedforward // nranks
        self._dim_feedforward = dim_feedforward
        self._dropout_rate = dropout_rate
        self._act_dropout_rate = (
            dropout_rate if act_dropout_rate is None else act_dropout_rate
        )
        self._seed = seed
        self._act_method = activation
        self._normalize_before = normalize_before
        self._epsilon = epsilon
        self._ring_id = ring_id

        self._linear1_weight = self.create_parameter(
            shape=[d_model, dim_feedforward],
            attr=linear1_weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        self._linear1_bias = self.create_parameter(
            shape=[dim_feedforward],
            attr=linear1_bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )

        self._linear2_weight = self.create_parameter(
            shape=[dim_feedforward, d_model],
            attr=linear2_weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )

        self._linear2_bias = self.create_parameter(
            shape=[d_model],
            attr=linear2_bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )

        if nranks > 1:
            assert ring_id != -1
            # column parallel
            _set_var_distributed(self._linear1_weight)
            _set_var_distributed(self._linear1_bias)
            _set_var_distributed(self._linear2_weight)

        if normalize_before:
            self._ln1_scale = self.create_parameter(
                shape=[d_model],
                attr=ln1_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
            )
            self._ln1_bias = self.create_parameter(
                shape=[d_model], attr=ln1_bias_attr, is_bias=True
            )
            self._ln2_scale = None
            self._ln2_bias = None
        else:
            self._ln1_scale = None
            self._ln1_bias = None
            self._ln2_scale = self.create_parameter(
                shape=[d_model],
                attr=ln2_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
            )
            self._ln2_bias = self.create_parameter(
                shape=[d_model], attr=ln2_bias_attr, is_bias=True
            )

        self.name = name

    def forward(self, src, cache=None):
        out = incubate_f.fused_feedforward(
            src,
            self._linear1_weight,
            self._linear2_weight,
            self._linear1_bias,
            self._linear2_bias,
            self._ln1_scale,
            self._ln1_bias,
            self._ln2_scale,
            self._ln2_bias,
            dropout1_rate=self._act_dropout_rate,
            dropout2_rate=self._dropout_rate,
            seed=self._seed,
            activation=self._act_method,
            ln1_epsilon=self._epsilon,
            ln2_epsilon=self._epsilon,
            pre_layer_norm=self._normalize_before,
            training=self.training,
            ring_id=self._ring_id,
            name=self.name,
        )
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'd_model={}, dim_feedforward={}, dropout_rate={}, epsilon={}, activation={}, act_dropout_rate={}, normalize_before={}, dtype={}{}'.format(
            self._d_model,
            self._dim_feedforward,
            self._dropout_rate,
            self._epsilon,
            self._act_method,
            self._act_dropout_rate,
            self._normalize_before,
            self._dtype,
            name_str,
        )

    def _amp_decorate(self, dtype):
        # tmp fix for amp.decorator(O2)
        layer_norm_params_id = []
        if self._normalize_before:
            layer_norm_params_id.append(id(self._ln1_scale))
            layer_norm_params_id.append(id(self._ln1_bias))
        else:
            layer_norm_params_id.append(id(self._ln2_scale))
            layer_norm_params_id.append(id(self._ln2_bias))

        for key, param in self._parameters.items():
            if id(param) in layer_norm_params_id:
                continue
            if param is not None:
                with no_grad():
                    param_applied = _to_dtype(param, dtype)

        self._dtype = dtype


class FusedTransformerEncoderLayer(Layer):
    """

    FusedTransformerEncoderLayer is composed of two sub-layers which are self (multi-head)
    attention and feedforward network. Before and after each sub-layer, pre-process
    and post-precess would be applied on the input and output accordingly. If
    `normalize_before` is True, pre-process is layer normalization and post-precess
    includes dropout, residual connection. Otherwise, no pre-process and post-precess
    includes dropout, residual connection, layer normalization.

    Parameters:
        d_model (int): The expected feature size in the input and output.
        nhead (int): The number of heads in multi-head attention(MHA).
        dim_feedforward (int): The hidden layer size in the feedforward network(FFN).
        dropout_rate (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.1
        activation (str, optional): The activation function in the feedforward
            network. Default relu.
        attn_dropout_rate (float, optional): The dropout probability used
            in MHA to drop some attention target. If None, use the value of
            `dropout`. Default None
        act_dropout_rate (float, optional): The dropout probability used after FFN
            activition.  If None, use the value of `dropout`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into preprocessing of MHA and FFN sub-layers. If True, pre-process is layer
            normalization and post-precess includes dropout, residual connection.
            Otherwise, no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Default False
        weight_attr(ParamAttr|list|tuple, optional): To specify the weight parameter property.
            If it is a list/tuple, `weight_attr[0]` would be used as `weight_attr` for
            MHA, and `weight_attr[1]` would be used as `weight_attr` for linear in FFN.
            Otherwise, MHA and FFN both use it as `weight_attr` to create parameters.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :code:`ParamAttr` .
        bias_attr (ParamAttr|list|tuple|bool, optional): To specify the bias parameter property.
            If it is a list/tuple, `bias_attr[0]` would be used as `bias_attr` for
            MHA, and `bias_attr[1]` would be used as `bias_attr` for linear in FFN.
            Otherwise, MHA and FFN both use it as `bias_attr` to create parameters.
            The `False` value means the corresponding layer would not have trainable
            bias parameter. See usage for details in :code:`ParamAttr` . Default: None,
            which means the default bias parameter property is used.


    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn import FusedTransformerEncoderLayer

            # encoder input: [batch_size, src_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, n_head, src_len, src_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            encoder_layer = FusedTransformerEncoderLayer(128, 2, 512)
            enc_output = encoder_layer(enc_input, attn_mask)  # [2, 4, 128]

    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout_rate=0.1,
        activation="relu",
        attn_dropout_rate=None,
        act_dropout_rate=None,
        normalize_before=False,
        weight_attr=None,
        bias_attr=None,
    ):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(FusedTransformerEncoderLayer, self).__init__()
        assert (
            d_model > 0
        ), "Expected d_model to be greater than 0, " "but received {}".format(
            d_model
        )
        assert (
            nhead > 0
        ), "Expected nhead to be greater than 0, " "but received {}".format(
            nhead
        )
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, "
            "but received {}".format(dim_feedforward)
        )
        attn_dropout_rate = (
            dropout_rate if attn_dropout_rate is None else attn_dropout_rate
        )
        act_dropout_rate = (
            dropout_rate if act_dropout_rate is None else act_dropout_rate
        )
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.fused_attn = FusedMultiHeadAttention(
            d_model,
            nhead,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            normalize_before=self.normalize_before,
            qkv_weight_attr=weight_attrs[0],
            qkv_bias_attr=bias_attrs[0],
            linear_weight_attr=weight_attrs[0],
            linear_bias_attr=bias_attrs[0],
            pre_ln_scale_attr=weight_attrs[0],
            pre_ln_bias_attr=bias_attrs[0],
            ln_scale_attr=weight_attrs[0],
            ln_bias_attr=bias_attrs[0],
        )

        self.ffn = FusedFeedForward(
            d_model,
            dim_feedforward,
            dropout_rate=dropout_rate,
            activation=activation,
            act_dropout_rate=act_dropout_rate,
            normalize_before=self.normalize_before,
            linear1_weight_attr=weight_attrs[1],
            linear1_bias_attr=bias_attrs[1],
            linear2_weight_attr=weight_attrs[1],
            linear2_bias_attr=bias_attrs[1],
        )

    def forward(self, src, src_mask=None, cache=None):
        """

        Applies a Transformer encoder layer on the input.

        Parameters:
            src (Tensor): The input of Transformer encoder layer. It is
                a tensor with shape `[batch_size, sequence_length, d_model]`.
                The data type should be float32 or float64.
            src_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
            cache (Tensor, optional): It is an instance of `MultiHeadAttention.Cache`.
                See :ref:`api_paddle_nn_TransformerEncoderLayer`.gen_cache for more details. It is
                only used for inference and should be None for training. Default
                None.

        Returns:
            Tensor|tuple, It is a tensor that has the same shape and data type \
                as `enc_input`, representing the output of Transformer encoder \
                layer. Or a tuple if `cache` is not None, except for encoder \
                layer output, the tuple includes the new cache which is same \
                as input `cache` argument but `incremental_cache` has an \
                incremental length. See `MultiHeadAttention.gen_cache` and \
                `MultiHeadAttention.forward` for more details.

        """
        src_mask = _convert_attention_mask(src_mask, src.dtype)
        if cache is None:
            attn_out = self.fused_attn(src, attn_mask=src_mask)
        else:
            attn_out, incremental_cache = self.fused_attn(
                src, attn_mask=src_mask, cache=cache
            )

        ffn_out = self.ffn(attn_out)

        return ffn_out if cache is None else (ffn_out, incremental_cache)


class FusedTransformer(Layer):
    """
    A Transformer model composed of an instance of `TransformerEncoder` and an
    instance of `TransformerDecoder`. While the embedding layer and output layer
    are not included.

    Please refer to `Attention is all you need <http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>`_ ,
    and see `TransformerEncoder` and `TransformerDecoder` for more details.

    Users can configurate the model architecture with corresponding parameters.
    Note the usage of `normalize_before` representing where to apply layer
    normalization (in pre-process or post-precess of multi-head attention or FFN),
    and some transformer like models are different on this, such as
    `BERT <https://arxiv.org/abs/1810.04805>`_ and `GPT2 <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`_ .
    The default architecture here places layer normalization in post-process and
    applies another layer normalization on the output of last encoder/decoder layer.

    Parameters:
        d_model (int, optional): The expected feature size in the encoder/decoder input
            and output. Default 512
        nhead (int, optional): The number of heads in multi-head attention(MHA). Default 8
        num_encoder_layers (int, optional): The number of layers in encoder. Default 6
        num_decoder_layers (int, optional): The number of layers in decoder. Default 6
        dim_feedforward (int, optional): The hidden layer size in the feedforward network(FFN). Default 2048
        dropout (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.1
        activation (str, optional): The activation function in the feedforward
            network. Default relu.
        attn_dropout (float, optional): The dropout probability used
            in MHA to drop some attention target. If None, use the value of
            `dropout`. Default None
        act_dropout (float, optional): The dropout probability used after FFN
            activition.  If None, use the value of `dropout`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into preprocessing of MHA and FFN sub-layers. If True, pre-process is layer
            normalization and post-precess includes dropout, residual connection.
            Otherwise, no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Default False
        weight_attr(ParamAttr|list|tuple, optional): To specify the weight parameter property.
            If it is a list/tuple, the length of `weight_attr` could be 1, 2 or 3. If it is 3,
            `weight_attr[0]` would be used as `weight_attr` for self attention, `weight_attr[1]`
            would be used as `weight_attr` for cross attention of `TransformerDecoder`,
            and `weight_attr[2]` would be used as `weight_attr` for linear in FFN.
            If it is 2, `weight_attr[0]` would be used as `weight_attr` both for self attention
            and cross attntion and `weight_attr[1]` would be used as `weight_attr` for
            linear in FFN. If it is 1, `weight_attr[0]` would be used as `weight_attr`
            for self attention, cross attention and linear in FFN. Otherwise,
            the three sub-layers all uses it as `weight_attr` to create parameters.
            Default: None, which means the default weight parameter property is used.
            See usage for details
            in :code:`ParamAttr` .
        bias_attr (ParamAttr|list|tuple|bool, optional): To specify the bias parameter property.
            If it is a list/tuple, the length of `bias_attr` could be 1, 2 or 3. If it is 3,
            `bias_attr[0]` would be used as `bias_attr` for self attention, `bias_attr[1]`
            would be used as `bias_attr` for cross attention of `TransformerDecoder`,
            and `bias_attr[2]` would be used as `bias_attr` for linear in FFN.
            If it is 2, `bias_attr[0]` would be used as `bias_attr` both for self attention
            and cross attntion and `bias_attr[1]` would be used as `bias_attr` for
            linear in FFN. If it is 1, `bias_attr[0]` would be used as `bias_attr`
            for self attention, cross attention and linear in FFN. Otherwise,
            the three sub-layers all uses it as `bias_attr` to create parameters.
            The `False` value means the corresponding layer would not have trainable
            bias parameter. See usage for details in :code:`ParamAttr` .
            Default: None,which means the default bias parameter property is used.
        custom_encoder (Layer, optional): If custom encoder is provided, use it as the encoder.
            Default None
        custom_decoder (Layer, optional): If custom decoder is provided, use it as the decoder.
            Default None

    Examples:

        .. code-block:: python

            import paddle
            from paddle.nn import Transformer

            # src: [batch_size, tgt_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # tgt: [batch_size, src_len, d_model]
            dec_input = paddle.rand((2, 6, 128))
            # src_mask: [batch_size, n_head, src_len, src_len]
            enc_self_attn_mask = paddle.rand((2, 2, 4, 4))
            # tgt_mask: [batch_size, n_head, tgt_len, tgt_len]
            dec_self_attn_mask = paddle.rand((2, 2, 6, 6))
            # memory_mask: [batch_size, n_head, tgt_len, src_len]
            cross_attn_mask = paddle.rand((2, 2, 6, 4))
            transformer = Transformer(128, 2, 4, 4, 512)
            output = transformer(enc_input,
                                 dec_input,
                                 enc_self_attn_mask,
                                 dec_self_attn_mask,
                                 cross_attn_mask)  # [2, 6, 128]
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        attn_dropout=None,
        act_dropout=None,
        normalize_before=False,
        weight_attr=None,
        bias_attr=None,
        custom_encoder=None,
        custom_decoder=None,
    ):
        super(fusedTransformer, self).__init__()
        raise NotImplementedError()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        raise NotImplementedError()


class FusedMultiTransformer(Layer):
    """
    FusedMultiTransformer is composed of multi transformer layers which contains two
    sub-layers which are self (multi-head) attention and feedforward network. The
    function of one transformer layer is consistent with the following pseudo code:

    .. code-block:: python

        if pre_layer_norm:
            out = layer_norm(x)
            out = qkv_linear(out) + qkv_bias
        else:
            out = qkv_linear(x) + qkv_bias
        out = transpose(out, perm=[2, 0, 3, 1, 4])
        # extract q, k and v from out.
        q = out[0:1, ::]
        k = out[1:2, ::]
        v = out[2:3, ::]
        out = q * k^t
        out = attn_mask + out
        out = softmax(out)
        out = dropout(out)
        out = out * v
        out = transpose(out, perm=[0, 2, 1, 3])
        out = linear(out)
        if pre_layer_norm:
            out = x + dropout(out + bias)
        else:
            out = layer_norm(x + dropout(out + bias))

        residual = out;
        if pre_layer_norm:
            out = ffn_layer_norm(out)
        out = ffn1_linear(out)
        out = dropout(activation(out + ffn1_bias))
        out = ffn2_linear(out)
        out = residual + dropout(out + ffn2_bias)
        if not pre_layer_norm:
            out = ffn_layer_norm(out)

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention(MHA).
        dim_feedforward (int): The hidden layer size in the feedforward network(FFN).
        dropout_rate (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.0
        activation (str, optional): The activation function in the feedforward
            network. Default "gelu".
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into preprocessing of MHA and FFN sub-layers. If True, pre-process is layer
            normalization and post-precess includes dropout, residual connection.
            Otherwise, no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Default True
        ln_scale_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for Attention layer_norm. For Attention layer_norm weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for Attention layer_norm. For Attention layer_norm bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        qkv_weight_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for Attention qkv computation. For Attention qkv weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        qkv_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for Attention qkv computation. For Attention qkv bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear_weight_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for Attention linear. For Attention linear weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for Attention linear computation. For Attention linear bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn_ln_scale_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for FFN layer_norm. For FFN layer_norm weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn_ln_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for FFN layer_norm. For FFN layer_norm bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn1_weight_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for FFN first linear. For FFN first linear weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn1_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for FFN first linear. For FFN first linear bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn2_weight_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for FFN second linear. For FFN second linear weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn2_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for FFN second linear. For FFN second linear bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        epsilon (float, optional): Small float value added to denominator of the layer_norm to
            avoid dividing by zero. Default: 1e-05.
        num_layers (int, optional): The number of layers of the transformer. If `qkv_weight_attrs`
            is a list or tuple, the number of layers is obtained from `qkv_weight_attrs`. num_layers
            only takes effect when `qkv_weight_attrs` is not a list or tuple. Default: -1.
        nranks (int, optional): Distributed tensor model parallel nranks. Default is 1, means not using mp.
        trans_qkvw (bool, optional): Whether to transpose for weights of qkv.
            If true, the shape eights of qkv should be [3, num_head, dim_head, dim_embed].
            Otherwise the shape of weights of qkv should be [dim_embed, 3, num_head, dim_head]. Default: True.
        ring_id (int, optional): For distributed tensor model parallel. Default is -1, means not using mp.
        name (str, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Examples:

        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn import FusedMultiTransformer

            # encoder input: [batch_size, src_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, 1, src_len, src_len]
            attn_mask = paddle.rand((2, 1, 4, 4))
            encoder_layers = FusedMultiTransformer(128, 2, 512, num_layers=1)
            enc_output = encoder_layers(enc_input, attn_mask)  # [2, 4, 128]
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward,
        dropout_rate=0.0,
        activation="gelu",
        normalize_before=True,
        ln_scale_attrs=None,
        ln_bias_attrs=None,
        qkv_weight_attrs=None,
        qkv_bias_attrs=None,
        linear_weight_attrs=None,
        linear_bias_attrs=None,
        ffn_ln_scale_attrs=None,
        ffn_ln_bias_attrs=None,
        ffn1_weight_attrs=None,
        ffn1_bias_attrs=None,
        ffn2_weight_attrs=None,
        ffn2_bias_attrs=None,
        epsilon=1e-5,
        num_layers=-1,
        nranks=1,
        trans_qkvw=True,
        ring_id=-1,
        name=None,
        dy_to_st=False,
    ):
        super(FusedMultiTransformer, self).__init__()

        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            "but received {}".format(embed_dim)
        )
        assert (
            num_heads > 0
        ), "Expected nhead to be greater than 0, " "but received {}".format(
            num_heads
        )
        assert (
            dim_feedforward > 0
        ), "Expected dim_feedforward to be greater than 0, but received {}".format(
            dim_feedforward
        )

        self.normalize_before = normalize_before
        self._dtype = "float16"
        #self._dtype = self._helper.get_default_dtype()
        self._epsilon = epsilon
        self._trans_qkvw = trans_qkvw
        self._ring_id = ring_id

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
        assert num_heads % nranks == 0
        assert dim_feedforward % nranks == 0
        num_heads = num_heads // nranks
        dim_feedforward = dim_feedforward // nranks
        self._dim_feedforward = dim_feedforward

        if isinstance(qkv_weight_attrs, (list, tuple, ParameterList)):
            num_layers = len(qkv_weight_attrs)
        assert num_layers > 0

        # if not dy_to_st:
        #     self.ln_scales, self.ln_biases = [], []
        #     self.qkv_weights, self.qkv_biases = [], []
        #     self.linear_weights, self.linear_biases = [], []
        #     self.ffn_ln_scales, self.ffn_ln_biases = [], []
        #     self.ffn1_weights, self.ffn1_biases = [], []
        #     self.ffn2_weights, self.ffn2_biases = [], []
        # else:
        self.ln_scales, self.ln_biases = ParameterList(), ParameterList()
        self.qkv_weights, self.qkv_biases = ParameterList(), ParameterList()
        self.linear_weights, self.linear_biases = ParameterList(), ParameterList()
        self.ffn_ln_scales, self.ffn_ln_biases = ParameterList(), ParameterList()
        self.ffn1_weights, self.ffn1_biases = ParameterList(), ParameterList()
        self.ffn2_weights, self.ffn2_biases = ParameterList(), ParameterList()
        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple, ParameterList)):
                assert len(attrs) == num_layers
                return attrs[idx]
            return attrs

        for i in range(num_layers):
            ln_scale_attr = get_attr(ln_scale_attrs, i)
            ln_bias_attr = get_attr(ln_bias_attrs, i)
            qkv_weight_attr = get_attr(qkv_weight_attrs, i)
            qkv_bias_attr = get_attr(qkv_bias_attrs, i)
            linear_weight_attr = get_attr(linear_weight_attrs, i)
            linear_bias_attr = get_attr(linear_bias_attrs, i)

            ffn_ln_scale_attr = get_attr(ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = get_attr(ffn_ln_bias_attrs, i)
            ffn1_weight_attr = get_attr(ffn1_weight_attrs, i)
            ffn1_bias_attr = get_attr(ffn1_bias_attrs, i)
            ffn2_weight_attr = get_attr(ffn2_weight_attrs, i)
            ffn2_bias_attr = get_attr(ffn2_bias_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
                dtype="float32",
            )
            ln_bias = self.create_parameter(
                attr=ln_bias_attr, shape=[embed_dim], is_bias=True, dtype="float32"
            )
            qkv_weight = self.create_parameter(
                shape=[3, num_heads, self.head_dim, embed_dim]
                if trans_qkvw
                else [embed_dim, 3, num_heads, self.head_dim],
                attr=qkv_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            qkv_bias = self.create_parameter(
                shape=[3, num_heads, self.head_dim],
                attr=qkv_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            linear_weight = self.create_parameter(
                shape=[num_heads * self.head_dim, embed_dim],
                attr=linear_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            linear_bias = self.create_parameter(
                shape=[embed_dim],
                attr=linear_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )

            ffn_ln_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
                dtype="float32",
            )
            ffn_ln_bias = self.create_parameter(
                shape=[embed_dim], attr=ffn_ln_bias_attr, is_bias=True, dtype="float32"
            )
            ffn1_weight = self.create_parameter(
                shape=[embed_dim, dim_feedforward],
                attr=ffn1_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            ffn1_bias = self.create_parameter(
                shape=[dim_feedforward],
                attr=ffn1_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            ffn2_weight = self.create_parameter(
                shape=[dim_feedforward, embed_dim],
                attr=ffn2_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            ffn2_bias = self.create_parameter(
                shape=[embed_dim],
                attr=ffn2_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )

            # tensor model parallel
            if nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(qkv_bias)
                _set_var_distributed(ffn1_weight)
                _set_var_distributed(ffn1_bias)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(ffn2_weight)

            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            self.qkv_weights.append(qkv_weight)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_biases.append(linear_bias)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.ffn1_weights.append(ffn1_weight)
            self.ffn1_biases.append(ffn1_bias)
            self.ffn2_weights.append(ffn2_weight)
            self.ffn2_biases.append(ffn2_bias)

        self.dropout_rate = dropout_rate
        self.activation = activation
        self.name = name

    def forward(self, src, attn_mask=None, caches=None, seq_lens=None, beam_offset=None, time_step=None):
        """
        Applies multi transformer layers on the input.

        Parameters:
            src (Tensor): The input of Transformer layers. It is
                a tensor with shape `[batch_size, sequence_length, d_model]`.
                The data type should be float16 or float32.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                `[batch_size, 1, sequence_length, sequence_length]`. It can be
                None when nothing wanted or needed to be prevented attention to.
                Default None.
            caches (list(Tensor)|tuple(Tensor), optional): The cache structure
                tensors for the inference generation model. It is only used for
                inference and should be None for training. The shape is
                `[2, batch_size, num_head, max_seq_len, head_dim]`. Default None.
            time_step (Tensor, optional): The time step tensor for the generation
                model. Which used in decode stage, to represent the time step,
                that is, the real seq_len of CacheKV. The shape is `[1]`, must be
                in CPUPlace. Default None.

        Returns:
            Tensor|tuple: If `caches` is None, return a tensor that has
            the same shape and data type with `src`, representing the output
            of Transformer layers. If `caches` is not None, return the
            tuple (output, caches), which output is the output of
            Transformer layers, caches is inplace with input `caches`.
        """

        if caches is not None:
            assert len(caches) == len(self.qkv_weights)
        out = incubate_f.fused_multi_transformer(
            src,
            self.ln_scales,
            self.ln_biases,
            self.qkv_weights,
            self.qkv_biases,
            self.linear_weights,
            self.linear_biases,
            self.ffn_ln_scales,
            self.ffn_ln_biases,
            self.ffn1_weights,
            self.ffn1_biases,
            self.ffn2_weights,
            self.ffn2_biases,
            pre_layer_norm=self.normalize_before,
            epsilon=self._epsilon,
            cache_kvs=caches,
            beam_offset=beam_offset,
            time_step=time_step,
            seq_lens=seq_lens,
            attn_mask=attn_mask,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            training=self.training,
            mode='upscale_in_train',
            trans_qkvw=self._trans_qkvw,
            ring_id=self._ring_id,
            name=self.name,
        )
        return out

    def _amp_decorate(self, dtype):
        # tmp fix for amp.decorator(O2)
        def trans_to_fp16(l):
            for param in l:
                if param is not None:
                    with no_grad():
                        param_applied = _to_dtype(param, dtype)
        trans_to_fp16(self.qkv_weights)
        trans_to_fp16(self.qkv_biases)
        trans_to_fp16(self.linear_weights)
        trans_to_fp16(self.linear_biases)
        trans_to_fp16(self.ffn1_weights)
        trans_to_fp16(self.ffn1_biases)
        trans_to_fp16(self.ffn2_weights)
        trans_to_fp16(self.ffn2_biases)
        self._dtype = dtype

class FusedMultiTransformerWeightOnly(Layer):
    """
    FusedMultiTransfor on weight quant
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward,
        dropout_rate=0.0,
        activation="gelu",
        weight_dtype="int8",
        normalize_before=True,
        ln_scale_attrs=None,
        ln_bias_attrs=None,
        qkv_weight_attrs=None,
        qkv_scale_attrs=None,
        qkv_bias_attrs=None,
        linear_weight_attrs=None,
        linear_scale_attrs=None,
        linear_bias_attrs=None,
        ffn_ln_scale_attrs=None,
        ffn_ln_bias_attrs=None,
        ffn1_weight_attrs=None,
        ffn1_scale_attrs=None,
        ffn1_bias_attrs=None,
        ffn2_weight_attrs=None,
        ffn2_scale_attrs=None,
        ffn2_bias_attrs=None,
        epsilon=1e-5,
        num_layers=-1,
        nranks=1,
        ring_id=-1,
        name=None,
        dy_to_st=False,
    ):
        super(FusedMultiTransformerWeightOnly, self).__init__()

        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            "but received {}".format(embed_dim)
        )
        assert (
            num_heads > 0
        ), "Expected nhead to be greater than 0, " "but received {}".format(
            num_heads
        )
        assert (
            dim_feedforward > 0
        ), "Expected dim_feedforward to be greater than 0, but received {}".format(
            dim_feedforward
        )

        self.normalize_before = normalize_before
        #self._dtype = self._helper.get_default_dtype()
        self._dtype = "float16"
        self._epsilon = epsilon
        self._ring_id = ring_id

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
        assert num_heads % nranks == 0
        assert dim_feedforward % nranks == 0
        num_heads = num_heads // nranks
        #dim_feedforward = dim_feedforward // nranks
        self._dim_feedforward = dim_feedforward
        self._weight_dtype = weight_dtype

        if isinstance(qkv_weight_attrs, (list, tuple, ParameterList)):
            num_layers = len(qkv_weight_attrs)
        assert num_layers > 0

        self.ln_scales, self.ln_biases = ParameterList(), ParameterList()
        self.qkv_weights, self.qkv_scales, self.qkv_biases = ParameterList(), ParameterList(), ParameterList()
        #self.qkv_weights, self.qkv_biases = ParameterList(), ParameterList()
        self.linear_weights, self.linear_scales, self.linear_biases = ParameterList(), ParameterList(), ParameterList()
        #self.linear_weights, self.linear_biases = ParameterList(), ParameterList()
        self.ffn_ln_scales, self.ffn_ln_biases = ParameterList(), ParameterList()
        self.ffn1_weights, self.ffn1_scales, self.ffn1_biases = ParameterList(), ParameterList(), ParameterList()
        #self.ffn1_weights, self.ffn1_biases = ParameterList(), ParameterList()
        self.ffn2_weights, self.ffn2_scales, self.ffn2_biases = ParameterList(), ParameterList(), ParameterList()
        #self.ffn2_weights, self.ffn2_biases = ParameterList(), ParameterList()
        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple, ParameterList)):
                assert len(attrs) == num_layers
                return attrs[idx]
            return attrs
        weight_int8 = False if self._weight_dtype == "int4" else True
        print(f"_weight_dtype: {self._weight_dtype}, weight_int8: {weight_int8}")

        for i in range(num_layers):
            ln_scale_attr = get_attr(ln_scale_attrs, i)
            ln_bias_attr = get_attr(ln_bias_attrs, i)
            qkv_weight_attr = get_attr(qkv_weight_attrs, i)
            qkv_scale_attr = get_attr(qkv_scale_attrs, i)
            qkv_bias_attr = get_attr(qkv_bias_attrs, i)
            linear_weight_attr = get_attr(linear_weight_attrs, i)
            linear_scale_attr = get_attr(linear_scale_attrs, i)
            linear_bias_attr = get_attr(linear_bias_attrs, i)

            ffn_ln_scale_attr = get_attr(ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = get_attr(ffn_ln_bias_attrs, i)
            ffn1_weight_attr = get_attr(ffn1_weight_attrs, i)
            ffn1_scale_attr = get_attr(ffn1_scale_attrs, i)
            ffn1_bias_attr = get_attr(ffn1_bias_attrs, i)
            ffn2_weight_attr = get_attr(ffn2_weight_attrs, i)
            ffn2_scale_attr = get_attr(ffn2_scale_attrs, i)
            ffn2_bias_attr = get_attr(ffn2_bias_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
                dtype="float32",
            )
            ln_bias = self.create_parameter(
                attr=ln_bias_attr, shape=[embed_dim], is_bias=True, dtype="float32"
            )
            qkv_weight = self.create_parameter(
                shape=[3, num_heads, self.head_dim, embed_dim],
                attr=qkv_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            qkv_scale = self.create_parameter(
                shape=[int(3 * num_heads * self.head_dim)],
                attr=qkv_scale_attr,
                dtype=self._dtype,
                is_bias=False,
                default_initializer=Constant(value=1.0),
            )
            qkv_bias = self.create_parameter(
                shape=[3, num_heads, self.head_dim],
                attr=qkv_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            '''
            linear_weight = self.create_parameter(
                shape=[int(num_heads * self.head_dim), embed_dim],
                attr=linear_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            '''
            linear_weight = self.create_parameter(
                shape=[embed_dim if weight_int8 else int(embed_dim / 2),
                        int(num_heads * self.head_dim)],
                attr=linear_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            linear_scale = self.create_parameter(
                shape=[embed_dim],
                attr=linear_scale_attr,
                dtype=self._dtype,
                is_bias=False,
                default_initializer=Constant(1.0),
            )
            linear_bias = self.create_parameter(
                shape=[embed_dim],
                attr=linear_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            ffn_ln_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(value=1.0),
                dtype="float32",
            )
            ffn_ln_bias = self.create_parameter(
                shape=[embed_dim], attr=ffn_ln_bias_attr, is_bias=True, dtype="float32"
            )
            '''
            ffn1_weight = self.create_parameter(
                 shape=[embed_dim, dim_feedforward],
                 attr=ffn1_weight_attr,
                 dtype=self._dtype,
                 is_bias=False,
            )
            '''
            ffn1_weight = self.create_parameter(
                shape=[dim_feedforward, embed_dim],
                attr=ffn1_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            ffn1_scale = self.create_parameter(
                shape=[dim_feedforward],
                attr=ffn1_scale_attr,
                dtype=self._dtype,
                is_bias=False,
                default_initializer=Constant(value=1.0),
            )
            ffn1_bias = self.create_parameter(
                shape=[dim_feedforward],
                attr=ffn1_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            '''
            ffn2_weight = self.create_parameter(
                shape=[dim_feedforward, embed_dim],
                attr=ffn2_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            '''
            ffn2_weight = self.create_parameter(
                shape=[embed_dim, dim_feedforward],
                attr=ffn2_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            ffn2_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn2_scale_attr,
                dtype=self._dtype,
                is_bias=False,
                default_initializer=Constant(value=1.0),
            )
            ffn2_bias = self.create_parameter(
                shape=[embed_dim],
                attr=ffn2_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )

            # tensor model parallel
            if nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(qkv_bias)
                _set_var_distributed(ffn1_weight)
                _set_var_distributed(ffn1_bias)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(ffn2_weight)

            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            self.qkv_weights.append(qkv_weight)
            self.qkv_scales.append(qkv_scale)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_scales.append(linear_scale)
            self.linear_biases.append(linear_bias)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.ffn1_weights.append(ffn1_weight)
            self.ffn1_scales.append(ffn1_scale)
            self.ffn1_biases.append(ffn1_bias)
            self.ffn2_weights.append(ffn2_weight)
            self.ffn2_scales.append(ffn2_scale)
            self.ffn2_biases.append(ffn2_bias)

        self.dropout_rate = dropout_rate
        self.activation = activation
        self.name = name
        #trans weight to int8
        self._int8_decorate()


    def forward(self, src, attn_mask=None, caches=None, seq_lens=None, beam_offset=None, time_step=None):
        """
        Applies multi transformer weight only layers on the input.
        """
        cache_kv_out, final_out = _legacy_C_ops.fused_multi_transformer_weight_only(
            src,
            list(self.ln_scales),
            list(self.ln_biases),
            list(self.qkv_weights),
            list(self.qkv_scales),
            list(self.qkv_biases),
            caches,
            beam_offset,
            time_step,
            seq_lens,
            attn_mask,
            list(self.linear_weights),
            list(self.linear_scales),
            list(self.linear_biases),
            list(self.ffn_ln_scales),
            list(self.ffn_ln_biases),
            list(self.ffn1_weights),
            list(self.ffn1_scales),
            list(self.ffn1_biases),
            list(self.ffn2_weights),
            list(self.ffn2_scales),
            list(self.ffn2_biases),
            caches,
            'pre_layer_norm',
            self.normalize_before,
            'epsilon',
            self._epsilon,
            'dropout_rate',
            self.dropout_rate,
            'is_test',
            not self.training,
            'dropout_implementation',
            'upscale_in_train',
            'act_method',
            self.activation,
            'weight_dtype',
            self._weight_dtype,
            'ring_id',
            self._ring_id
        )
        if caches is not None:
            return final_out, cache_kv_out
        return final_out

    def _int8_decorate(self):
        # tmp fix for amp.decorator(O2)
        def trans_to_int8(l):
            for param in l:
                if param is not None:
                    with no_grad():
                        param_applied = _to_dtype(param, "int8")
        trans_to_int8(self.qkv_weights)
        trans_to_int8(self.linear_weights)
        trans_to_int8(self.ffn1_weights)
        trans_to_int8(self.ffn2_weights)
        self._dtype = "int8"



class FusedMultiTransformerINT8(Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dim_feedforward,
                 dropout_rate=0.0,
                 activation="gelu",
                 normalize_before=True,
                 ln_scale_attrs=None,
                 ln_bias_attrs=None,
                 qkv_weight_attrs=None,
                 qkv_bias_attrs=None,
                 linear_weight_attrs=None,
                 linear_bias_attrs=None,
                 ffn_ln_scale_attrs=None,
                 ffn_ln_bias_attrs=None,
                 ffn1_weight_attrs=None,
                 ffn1_bias_attrs=None,
                 ffn2_weight_attrs=None,
                 ffn2_bias_attrs=None,
                 qkv_out_scales_attrs=None,
                 out_linear_out_scales_attrs=None,
                 ffn1_out_scales_attrs=None,
                 ffn2_out_scales_attrs=None,
                 qkv_in_scale=None,
                 out_linear_in_scale=None,
                 ffn1_in_scale=None,
                 ffn2_in_scale=None,
                 epsilon=1e-5,
                 num_layers=-1,
                 nranks=1,
                 trans_qkvw=True,
                 ring_id=-1,
                 name=None):
        super(FusedMultiTransformerINT8, self).__init__()

        assert embed_dim > 0, ("Expected embed_dim to be greater than 0, "
                               "but received {}".format(embed_dim))
        assert num_heads > 0, ("Expected nhead to be greater than 0, "
                               "but received {}".format(num_heads))
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, but received {}".
            format(dim_feedforward))

        self.normalize_before = normalize_before
        # self._dtype = self._helper.get_default_dtype()
        self._dtype = "float16" # fix, default is fp16
        self._epsilon = epsilon
        self._trans_qkvw = trans_qkvw
        self._ring_id = ring_id

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_in_scale = qkv_in_scale
        self.out_linear_in_scale = out_linear_in_scale
        self.ffn1_in_scale = ffn1_in_scale
        self.ffn2_in_scale = ffn2_in_scale

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
        assert num_heads % nranks == 0
        assert dim_feedforward % nranks == 0
        num_heads = num_heads // nranks
        dim_feedforward = dim_feedforward // nranks
        self._dim_feedforward = dim_feedforward

        if isinstance(qkv_weight_attrs, (list, tuple)):
            num_layers = len(qkv_weight_attrs)
        assert num_layers > 0

        self.ln_scales, self.ln_biases = ParameterList(), ParameterList()
        self.qkv_weights, self.qkv_biases = ParameterList(), ParameterList()
        self.linear_weights, self.linear_biases = ParameterList(), ParameterList()
        self.ffn_ln_scales, self.ffn_ln_biases = ParameterList(), ParameterList()
        self.ffn1_weights, self.ffn1_biases = ParameterList(), ParameterList()
        self.ffn2_weights, self.ffn2_biases = ParameterList(), ParameterList()

        self.qkv_out_scales = ParameterList()
        self.out_linear_out_scales = ParameterList()
        self.ffn1_out_scales = ParameterList()
        self.ffn2_out_scales = ParameterList()

        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple)):
                assert len(attrs) == num_layers
                return attrs[idx]
            return attrs

        for i in range(num_layers):
            ln_scale_attr = get_attr(ln_scale_attrs, i)
            ln_bias_attr = get_attr(ln_bias_attrs, i)
            qkv_weight_attr = get_attr(qkv_weight_attrs, i)
            qkv_bias_attr = get_attr(qkv_bias_attrs, i)
            linear_weight_attr = get_attr(linear_weight_attrs, i)
            linear_bias_attr = get_attr(linear_bias_attrs, i)

            ffn_ln_scale_attr = get_attr(ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = get_attr(ffn_ln_bias_attrs, i)
            ffn1_weight_attr = get_attr(ffn1_weight_attrs, i)
            ffn1_bias_attr = get_attr(ffn1_bias_attrs, i)
            ffn2_weight_attr = get_attr(ffn2_weight_attrs, i)
            ffn2_bias_attr = get_attr(ffn2_bias_attrs, i)

            qkv_out_scales_attr = get_attr(qkv_out_scales_attrs, i)
            out_linear_out_scales_attr = get_attr(out_linear_out_scales_attrs, i)
            ffn1_out_scales_attr = get_attr(ffn1_out_scales_attrs, i)
            ffn2_out_scales_attr = get_attr(ffn2_out_scales_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
                dtype="float32")
            ln_bias = self.create_parameter(attr=ln_bias_attr,
                                            shape=[embed_dim],
                                            is_bias=True,
                                            dtype="float32")
            qkv_weight = self.create_parameter(
                shape=[3, num_heads, self.head_dim, embed_dim]
                if trans_qkvw else [embed_dim, 3, num_heads, self.head_dim],
                attr=qkv_weight_attr,
                dtype=self._dtype,
                is_bias=False)
            qkv_bias = self.create_parameter(
                shape=[3, num_heads, self.head_dim],
                attr=qkv_bias_attr,
                dtype=self._dtype,
                is_bias=True)
            linear_weight = self.create_parameter(
                shape=[num_heads * self.head_dim, embed_dim],
                attr=linear_weight_attr,
                dtype=self._dtype,
                is_bias=False)
            linear_bias = self.create_parameter(shape=[embed_dim],
                                                attr=linear_bias_attr,
                                                dtype=self._dtype,
                                                is_bias=True)

            ffn_ln_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
                dtype="float32")
            ffn_ln_bias = self.create_parameter(shape=[embed_dim],
                                                attr=ffn_ln_bias_attr,
                                                is_bias=True,
                                                dtype="float32")
            ffn1_weight = self.create_parameter(
                # shape=[embed_dim, dim_feedforward],
                shape=[dim_feedforward, embed_dim],
                attr=ffn1_weight_attr,
                dtype=self._dtype,
                is_bias=False)
            ffn1_bias = self.create_parameter(shape=[dim_feedforward],
                                              attr=ffn1_bias_attr,
                                              dtype=self._dtype,
                                              is_bias=True)
            ffn2_weight = self.create_parameter(
                # shape=[dim_feedforward, embed_dim],
                shape=[embed_dim, dim_feedforward],
                attr=ffn2_weight_attr,
                dtype=self._dtype,
                is_bias=False)
            ffn2_bias = self.create_parameter(shape=[embed_dim],
                                              attr=ffn2_bias_attr,
                                              dtype=self._dtype,
                                              is_bias=True)

            qkv_out_scale = self.create_parameter(
                shape=[3 * embed_dim],
                attr=qkv_out_scales_attr,
                dtype="float32",
                is_bias=False)
            out_linear_out_scale = self.create_parameter(
                shape=[embed_dim],
                attr=out_linear_out_scales_attr,
                dtype="float32",
                is_bias=False)
            ffn1_out_scale = self.create_parameter(
                shape=[4 * embed_dim],
                attr=ffn1_out_scales_attr,
                dtype="float32",
                is_bias=False)
            ffn2_out_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn2_out_scales_attr,
                dtype="float32",
                is_bias=False)

            # tensor model parallel
            if nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(qkv_bias)
                _set_var_distributed(ffn1_weight)
                _set_var_distributed(ffn1_bias)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(ffn2_weight)

            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            self.qkv_weights.append(qkv_weight)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_biases.append(linear_bias)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.ffn1_weights.append(ffn1_weight)
            self.ffn1_biases.append(ffn1_bias)
            self.ffn2_weights.append(ffn2_weight)
            self.ffn2_biases.append(ffn2_bias)

            self.qkv_out_scales.append(qkv_out_scale)
            self.out_linear_out_scales.append(out_linear_out_scale)
            self.ffn1_out_scales.append(ffn1_out_scale)
            self.ffn2_out_scales.append(ffn2_out_scale)

        self.dropout_rate = dropout_rate
        self.activation = activation
        self.name = name
        # int8 decorate
        self._int8_decorate()

    def forward(self, src, attn_mask=None, caches=None, seq_lens=None, beam_offset=None, time_step=None):
        """
        forward
        """
        cache_kv_out, final_out = _legacy_C_ops.fused_multi_transformer_int8(
            src,
            list(self.ln_scales),
            list(self.ln_biases),
            list(self.qkv_weights),
            list(self.qkv_biases),
            caches,
            beam_offset,
            time_step,
            seq_lens,
            attn_mask,
            list(self.linear_weights),
            list(self.linear_biases),
            list(self.ffn_ln_scales),
            list(self.ffn_ln_biases),
            list(self.ffn1_weights),
            list(self.ffn1_biases),
            list(self.ffn2_weights),
            list(self.ffn2_biases),
            list(self.qkv_out_scales),
            list(self.out_linear_out_scales),
            list(self.ffn1_out_scales),
            list(self.ffn2_out_scales),
            caches,
            'qkv_in_scale',
            self.qkv_in_scale,
            'out_linear_in_scale',
            self.out_linear_in_scale,
            'ffn1_in_scale',
            self.ffn1_in_scale,
            'ffn2_in_scale',
            self.ffn2_in_scale,
            'pre_layer_norm',
            self.normalize_before,
            'epsilon',
            self._epsilon,
            'dropout_rate',
            self.dropout_rate,
            'is_test',
            not self.training,
            'dropout_implementation',
            'upscale_in_train',
            'act_method',
            self.activation,
            'trans_qkvw',
            self._trans_qkvw,
            'ring_id',
            self._ring_id)

        if caches is not None:
            return final_out, cache_kv_out
        return final_out

    def _int8_decorate(self, dtype="int8"):
        # tmp fix for INT8
        def trans_to_int8(l):
            for param in l:
                if param is not None:
                    with no_grad():
                        param_applied = _to_dtype(param, dtype)
        trans_to_int8(self.qkv_weights)
        trans_to_int8(self.linear_weights)
        trans_to_int8(self.ffn1_weights)
        trans_to_int8(self.ffn2_weights)
        self._dtype = "int8"


class FusedMoELayer(Layer):
    """FusedMoE Layer
    Args:
        d_model: (int) model dimention
        num_expert: (int) expert count
        top_k: (int) top-k number
        some weights and bias...
        moe_group: moe group for experts communication
        mp_group: mp group for mp commutication
    Examples:
        .. code-block:: python
        # required: gpu
        import paddle
        from paddle.incubate.nn import FusedMoELayer

        # input: [batch_size, src_len, d_model]
        input = paddle.rand((2, 4, 128))
        # dim_feedforward = 128
        fused_moe_layer = FusedMoELayer(128, 128, 4, 2)
        output = fused_moe_layer(input)  # [2, 4, 128]

    """

    def __init__(self,
                 d_model,
                 dim_feedforward,
                 num_expert,
                 top_k,
                 approximate,
                 moe_group=None,
                 mp_group=None,
                 ln_scale=None,
                 ln_bias=None,
                 gate_weight=None,
                 gate_bias=None,
                 linear1_weights=None,
                 linear1_biases=None,
                 linear2_weights=None,
                 linear2_biases=None):
        super(FusedMoELayer, self).__init__()
        # only support mp/dp
        self.group = moe_group

        self.world_size = 1
        if self.group is not None:
            self.world_size = self.group.nranks
        self.num_expert = num_expert

        self.mp_group = mp_group
        self.mp_rank = 0
        self.mp_size = 1
        if mp_group is not None and mp_group.nranks > 1:
            self.mp_rank = mp_group.rank
            self.mp_size = mp_group.nranks
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.top_k = top_k
        self.approximate = approximate
        self.ln_scale = self.create_parameter(
                shape=[d_model],
                attr=None,
                is_bias=False
            )
        self.ln_bias = self.create_parameter(
            shape=[d_model], attr=None, is_bias=True
        )
        self.gate_weight = self.create_parameter(
                shape=[d_model, num_expert * self.world_size],
                attr=None,
                dtype=self._dtype,
                is_bias=False
            )
        self.gate_bias = self.create_parameter(
            shape=[num_expert * self.world_size],
            attr=None,
            dtype=self._dtype,
            is_bias=True
        )

        self.linear1_weights = ParameterList()
        self.linear2_weights = ParameterList()
        self.linear1_biases = ParameterList()
        self.linear2_biases = ParameterList()
        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple, ParameterList)):
                assert len(attrs) == num_expert
                return attrs[idx]
            return attrs
        for i in range(num_expert):
            w1 = get_attr(linear1_weights, i)
            b1 = get_attr(linear1_biases, i)
            w2 = get_attr(linear2_weights, i)
            b2 = get_attr(linear2_biases, i)

            self.linear1_weights.append(self.create_parameter(
                                            shape=[d_model, dim_feedforward],
                                            attr=w1,
                                            dtype=self._dtype,
                                            is_bias=False,
                                            default_initializer=nn.initializer.KaimingUniform()
            ))
            self.linear2_weights.append(self.create_parameter(
                                            shape=[dim_feedforward, d_model],
                                            attr=w2,
                                            dtype=self._dtype,
                                            is_bias=False,
                                            default_initializer=nn.initializer.KaimingUniform()
            ))
            self.linear1_biases.append(self.create_parameter(
                                            shape=[dim_feedforward],
                                            attr=b1,
                                            dtype=self._dtype,
                                            is_bias=True,
                                            default_initializer=nn.initializer.Constant(value=0.0)
            ))
            self.linear2_biases.append(self.create_parameter(
                                            shape=[d_model],
                                            attr=b2,
                                            dtype=self._dtype,
                                            is_bias=True,
                                            default_initializer=nn.initializer.Constant(value=0.0)
            ))
            self.linear1_weights[i].name = "expert_" + self.linear1_weights[i].name
            self.linear2_weights[i].name = "expert_" + self.linear2_weights[i].name
            self.linear1_biases[i].name = "expert_" + self.linear1_biases[i].name
            self.linear2_biases[i].name = "expert_" + self.linear2_biases[i].name

    def forward(self, inp):
        bsz = inp.shape[0]
        seq_len = inp.shape[1]
        out = _C_ops.fused_moe_kernel(
            inp,
            self.gate_weight,
            self.gate_bias,
            self.ln_scale,
            self.ln_bias,
            list(self.linear1_weights),
            list(self.linear1_biases),
            list(self.linear2_weights),
            list(self.linear2_biases),
            True,
            1e-5,
            self.top_k,
            self.mp_size,
            self.mp_rank,
            self.num_expert,
            self.world_size,
            -1 if self.group is None else self.group.id,
            self.approximate,
        )
        return out

    def _amp_decorate(self, dtype):
        # tmp fix for amp.decorator(O2)
        def trans_to_fp16(l):
            for param in l:
                if param is not None:
                    with paddle.no_grad():
                        param_applied = _to_dtype(param, dtype)
        trans_to_fp16(self.linear1_weights)
        trans_to_fp16(self.linear1_biases)
        trans_to_fp16(self.linear2_weights)
        trans_to_fp16(self.linear2_biases)
        _ = _to_dtype(self.gate_weight, dtype)
        _ = _to_dtype(self.gate_bias, dtype)
        self._dtype = dtype


class FusedMultiTransformerMoe(Layer):
    """
    FusedMultiTransformerMoe
    """
    def __init__(
        self,
        d_model,
        embed_dim,
        num_heads,
        dim_feedforward,
        dropout_rate=0.0,
        activation="gelu",
        normalize_before=True,
        ln_scale_attrs=None,
        ln_bias_attrs=None,
        qkv_weight_attrs=None,
        qkv_bias_attrs=None,
        linear_weight_attrs=None,
        linear_bias_attrs=None,
        gate_weight_attrs=None,
        gate_bias_attrs=None,
        ffn_ln_scale_attrs=None,
        ffn_ln_bias_attrs=None,
        expert_weight1_attrs=None,
        expert_bias1_attrs=None,
        expert_weight2_attrs=None,
        expert_bias2_attrs=None,
        epsilon=1e-5,
        num_layers=-1,
        nranks=1,
        trans_qkvw=True,
        ring_id=-1,
        num_expert=1,
        top_k=2,
        approximate=True,
        moe_group=None,
        mp_group=None,
        name=None,
    ):
        super(FusedMultiTransformerMoe, self).__init__()
        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            "but received {}".format(embed_dim)
        )
        assert (
            num_heads > 0
        ), "Expected nhead to be greater than 0, " "but received {}".format(
            num_heads
        )
        assert (
            dim_feedforward > 0
        ), "Expected dim_feedforward to be greater than 0, but received {}".format(
            dim_feedforward
        )
        gemm_cutlass = (os.getenv("FLAGS_enable_moe_gemm_cutlass", "false") == "true")
        if gemm_cutlass:
            print("FusedMultiTransformerMoe use cutlass gemm")
        # only support mp/dp
        # for moe config
        self.group = moe_group
        self.world_size = 1
        if self.group is not None:
            self.world_size = self.group.nranks
        self.num_expert = num_expert

        self.mp_rank = 0
        self.mp_size = 1
        if mp_group is not None and mp_group.nranks > 1:
            self.mp_rank = mp_group.rank
            self.mp_size = mp_group.nranks
        self.top_k = top_k
        self.approximate = approximate

        # origin fmt config
        self.normalize_before = normalize_before
        self._dtype = "float16"
        self._epsilon = epsilon
        self._trans_qkvw = trans_qkvw
        self._ring_id = ring_id

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
        assert num_heads % nranks == 0
        num_heads = num_heads // nranks

        if isinstance(qkv_weight_attrs, (list, tuple, ParameterList)):
            num_layers = len(qkv_weight_attrs)
        assert num_layers > 0

        self.ln_scales, self.ln_biases = ParameterList(), ParameterList()
        self.qkv_weights, self.qkv_biases = ParameterList(), ParameterList()
        self.linear_weights, self.linear_biases = ParameterList(), ParameterList()
        self.gate_weights, self.gate_biases = ParameterList(), ParameterList()
        self.ffn_ln_scales, self.ffn_ln_biases = ParameterList(), ParameterList()
        self.expert_weights1, self.expert_biases1 = ParameterList(), ParameterList()
        self.expert_weights2, self.expert_biases2 = ParameterList(), ParameterList()
        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple, ParameterList)):
                assert len(attrs) == num_layers
                return attrs[idx]
            return attrs

        for i in range(num_layers):
            ln_scale_attr = get_attr(ln_scale_attrs, i)
            ln_bias_attr = get_attr(ln_bias_attrs, i)
            qkv_weight_attr = get_attr(qkv_weight_attrs, i)
            qkv_bias_attr = get_attr(qkv_bias_attrs, i)
            linear_weight_attr = get_attr(linear_weight_attrs, i)
            linear_bias_attr = get_attr(linear_bias_attrs, i)

            ffn_ln_scale_attr = get_attr(ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = get_attr(ffn_ln_bias_attrs, i)
            gate_weight_attr = get_attr(gate_weight_attrs, i)
            gate_bias_attr = get_attr(gate_bias_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
                dtype="float32",
            )
            ln_bias = self.create_parameter(
                attr=ln_bias_attr, shape=[embed_dim], is_bias=True, dtype="float32"
            )
            qkv_weight = self.create_parameter(
                shape=[3, num_heads, self.head_dim, embed_dim]
                if trans_qkvw
                else [embed_dim, 3, num_heads, self.head_dim],
                attr=qkv_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            qkv_bias = self.create_parameter(
                shape=[3, num_heads, self.head_dim],
                attr=qkv_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            linear_weight = self.create_parameter(
                shape=[num_heads * self.head_dim, embed_dim],
                attr=linear_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            linear_bias = self.create_parameter(
                shape=[embed_dim],
                attr=linear_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )

            ffn_ln_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
                dtype="float32",
            )
            ffn_ln_bias = self.create_parameter(
                shape=[embed_dim], attr=ffn_ln_bias_attr, is_bias=True, dtype="float32"
            )
            gate_weight = self.create_parameter(
                shape=[d_model, num_expert * self.world_size],
                attr=gate_weight_attr,
                dtype=self._dtype,
                is_bias=False
            )
            gate_bias = self.create_parameter(
                shape=[num_expert * self.world_size],
                attr=gate_bias_attr,
                dtype=self._dtype,
                is_bias=True
            )

            # tensor model parallel
            if nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(qkv_bias)
                # row parallel
                _set_var_distributed(linear_weight)

            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            self.qkv_weights.append(qkv_weight)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_biases.append(linear_bias)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.gate_weights.append(gate_weight)
            self.gate_biases.append(gate_bias)

            for j in range(num_expert):
                expert_weight1_attr = get_attr(expert_weight1_attrs, i * num_expert + j)
                expert_bias1_attr = get_attr(expert_bias1_attrs, i * num_expert + j)
                expert_weight2_attr = get_attr(expert_weight2_attrs, i * num_expert + j)
                expert_bias2_attr = get_attr(expert_bias2_attrs, i * num_expert + j)

                expert_weight1 = self.create_parameter(
                    shape=[d_model, dim_feedforward] if not gemm_cutlass else [dim_feedforward, d_model],
                    attr=expert_weight1_attr,
                    dtype=self._dtype,
                    is_bias=False,
                    default_initializer=nn.initializer.KaimingUniform()
                )
                expert_bias1 = self.create_parameter(
                    shape=[dim_feedforward],
                    attr=expert_bias1_attr,
                    dtype=self._dtype,
                    is_bias=True,
                    default_initializer=nn.initializer.Constant(value=0.0)
                )
                expert_weight2 = self.create_parameter(
                    shape=[dim_feedforward, d_model] if not gemm_cutlass else [d_model, dim_feedforward],
                    attr=expert_weight2_attr,
                    dtype=self._dtype,
                    is_bias=False,
                    default_initializer=nn.initializer.KaimingUniform()
                )
                expert_bias2 = self.create_parameter(
                    shape=[d_model],
                    attr=expert_bias2_attr,
                    dtype=self._dtype,
                    is_bias=True,
                    default_initializer=nn.initializer.Constant(value=0.0)
                )
                expert_weight1.name = "expert_" + expert_weight1.name
                expert_bias1.name = "expert_" + expert_bias1.name
                expert_weight2.name = "expert_" + expert_weight2.name
                expert_bias2.name = "expert_" + expert_bias2.name
                self.expert_weights1.append(expert_weight1)
                self.expert_biases1.append(expert_bias1)
                self.expert_weights2.append(expert_weight2)
                self.expert_biases2.append(expert_bias2)
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.name = name
        if gemm_cutlass:
            self._share_expert_param(num_layers, num_expert, dim_feedforward, d_model)

    def forward(self, src, attn_mask=None, caches=None, seq_lens=None, beam_offset=None, time_step=None):
        """
        forward
        """
        cache_kv_out, final_out = _legacy_C_ops.fused_multi_transformer_moe(
            src,
            list(self.ln_scales),
            list(self.ln_biases),
            list(self.qkv_weights),
            list(self.qkv_biases),
            caches,
            beam_offset,
            time_step,
            seq_lens,
            attn_mask,
            list(self.linear_weights),
            list(self.linear_biases),
            list(self.gate_weights),
            list(self.gate_biases),
            list(self.ffn_ln_scales),
            list(self.ffn_ln_biases),
            list(self.expert_weights1),
            list(self.expert_biases1),
            list(self.expert_weights2),
            list(self.expert_biases2),
            caches,
            'pre_layer_norm',
            self.normalize_before,
            'epsilon',
            self._epsilon,
            'dropout_rate',
            self.dropout_rate,
            'is_test',
            not self.training,
            'dropout_implementation',
            'upscale_in_train',
            'act_method',
            self.activation,
            'trans_qkvw',
            self._trans_qkvw,
            'ring_id',
            self._ring_id,
            'topk',
            self.top_k,
            'mp_size',
            self.mp_size,
            'mp_rank',
            self.mp_rank,
            'num_expert',
            self.num_expert,
            'world_size',
            self.world_size,
            'moe_ring_id',
            -1 if self.group is None else self.group.id,
            'approximate',
            self.approximate
        )
        if caches is not None:
            return final_out, cache_kv_out
        return final_out

    def _amp_decorate(self, dtype):
        # tmp fix for amp.decorator(O2)
        def trans_to_fp16(l):
            for param in l:
                if param is not None:
                    with no_grad():
                        param_applied = _to_dtype(param, dtype)
        trans_to_fp16(self.qkv_weights)
        trans_to_fp16(self.qkv_biases)
        trans_to_fp16(self.linear_weights)
        trans_to_fp16(self.linear_biases)
        trans_to_fp16(self.gate_weights)
        trans_to_fp16(self.gate_biases)
        trans_to_fp16(self.expert_weights1)
        trans_to_fp16(self.expert_biases1)
        trans_to_fp16(self.expert_weights2)
        trans_to_fp16(self.expert_biases2)
        self._dtype = dtype

    def _share_expert_param(self, num_layers, num_expert, dim_feedforward, d_model):
        """
        share_param
        """
        def shard_tensor(dst_tensor, parent_tensor, pos):
            tmp = parent_tensor.value().get_tensor()._slice(pos, pos + 1)
            dst_tensor.value().get_tensor()._share_data_buffer(tmp, False)
            #print(dst_tensor)

        self.shared_weights1, self.shared_biases1 = ParameterList(), ParameterList()
        self.shared_weights2, self.shared_biases2 = ParameterList(), ParameterList()

        for i in range(num_layers):
            shared_weight1 = paddle.create_parameter(
                name=f"moe.expert.layer{i}.shared_weight1",
                shape=[num_expert, dim_feedforward, d_model],
                dtype=self._dtype,
                default_initializer=nn.initializer.Constant(value=0.0)) 
            shared_bias1 = paddle.create_parameter(
                name=f"moe.expert.layer{i}.shared_bias1",
                shape=[num_expert, dim_feedforward],
                dtype=self._dtype,
                default_initializer=nn.initializer.Constant(value=0.0)) 
            
            shared_weight2 = paddle.create_parameter(
                name=f"moe.expert.layer{i}.shared_weight2",
                shape=[num_expert, d_model, dim_feedforward],
                dtype=self._dtype,
                default_initializer=nn.initializer.Constant(value=0.0)) 
            shared_bias2 = paddle.create_parameter(
                name=f"moe.expert.layer{i}.shared_bias2",
                shape=[num_expert, d_model],
                dtype=self._dtype,
                default_initializer=nn.initializer.Constant(value=0.0)) 

            for j in range(self.num_expert):
                expert_idx = j + i * self.num_expert
                shard_tensor(self.expert_weights1[expert_idx], shared_weight1, j)
                shard_tensor(self.expert_biases1[expert_idx], shared_bias1, j)
                shard_tensor(self.expert_weights2[expert_idx], shared_weight2, j)
                shard_tensor(self.expert_biases2[expert_idx], shared_bias2, j)

            self.shared_weights1.append(shared_weight1)
            self.shared_biases1.append(shared_bias1)

            self.shared_weights2.append(shared_weight2)
            self.shared_biases2.append(shared_bias2)


class FusedMultiTransformerMoeINT8(Layer):
    """
    FusedMultiTransformerMoeINT8
    """
    def __init__(
        self,
        d_model,
        embed_dim,
        num_heads,
        dim_feedforward,
        dropout_rate=0.0,
        activation="gelu",
        normalize_before=True,
        ln_scale_attrs=None,
        ln_bias_attrs=None,
        qkv_weight_attrs=None,
        qkv_bias_attrs=None,
        linear_weight_attrs=None,
        linear_bias_attrs=None,
        gate_weight_attrs=None,
        gate_bias_attrs=None,
        ffn_ln_scale_attrs=None,
        ffn_ln_bias_attrs=None,
        expert_weight1_attrs=None,
        expert_bias1_attrs=None,
        expert_weight2_attrs=None,
        expert_bias2_attrs=None,
        qkv_out_scales_attrs=None, # out scales
        out_linear_out_scales_attrs=None,
        expert_weight1_out_scales_attrs=None,
        expert_weight2_out_scales_attrs=None,
        qkv_in_scale=None,
        out_linear_in_scale=None,
        expert_weight1_in_scale=None,
        expert_weight2_in_scale=None,
        epsilon=1e-5,
        num_layers=-1,
        nranks=1,
        trans_qkvw=True,
        ring_id=-1,
        num_expert=1,
        top_k=2,
        approximate=True,
        moe_group=None,
        mp_group=None,
        name=None,
    ):
        super(FusedMultiTransformerMoeINT8, self).__init__()
        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            "but received {}".format(embed_dim)
        )
        assert (
            num_heads > 0
        ), "Expected nhead to be greater than 0, " "but received {}".format(
            num_heads
        )
        assert (
            dim_feedforward > 0
        ), "Expected dim_feedforward to be greater than 0, but received {}".format(
            dim_feedforward
        )
        # only support mp/dp
        # for moe config
        self.group = moe_group
        self.world_size = 1
        if self.group is not None:
            self.world_size = self.group.nranks
        self.num_expert = num_expert

        self.mp_rank = 0
        self.mp_size = 1
        if mp_group is not None and mp_group.nranks > 1:
            self.mp_rank = mp_group.rank
            self.mp_size = mp_group.nranks
        self.top_k = top_k
        self.approximate = approximate

        # origin fmt config
        self.normalize_before = normalize_before
        # self._dtype = self._helper.get_default_dtype()
        self._dtype = "float16" # fix, default is fp16
        self._epsilon = epsilon
        self._trans_qkvw = trans_qkvw
        self._ring_id = ring_id

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
        assert num_heads % nranks == 0
        num_heads = num_heads // nranks

        if isinstance(qkv_weight_attrs, (list, tuple, ParameterList)):
            num_layers = len(qkv_weight_attrs)
        assert num_layers > 0

        self.qkv_in_scale = qkv_in_scale
        self.out_linear_in_scale = out_linear_in_scale
        self.expert_weight1_in_scale = expert_weight1_in_scale
        self.expert_weight2_in_scale = expert_weight2_in_scale

        self.ln_scales, self.ln_biases = ParameterList(), ParameterList()
        self.qkv_weights, self.qkv_biases = ParameterList(), ParameterList()
        self.linear_weights, self.linear_biases = ParameterList(), ParameterList()
        self.gate_weights, self.gate_biases = ParameterList(), ParameterList()
        self.ffn_ln_scales, self.ffn_ln_biases = ParameterList(), ParameterList()
        self.expert_weights1, self.expert_biases1 = ParameterList(), ParameterList()
        self.expert_weights2, self.expert_biases2 = ParameterList(), ParameterList()
        self.qkv_out_scales, self.out_linear_out_scales = ParameterList(), ParameterList()
        self.expert_weight1_out_scales, self.expert_weight2_out_scales = ParameterList(), ParameterList()
        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple, ParameterList)):
                assert len(attrs) == num_layers
                return attrs[idx]
            return attrs

        for i in range(num_layers):
            ln_scale_attr = get_attr(ln_scale_attrs, i)
            ln_bias_attr = get_attr(ln_bias_attrs, i)
            qkv_weight_attr = get_attr(qkv_weight_attrs, i)
            qkv_bias_attr = get_attr(qkv_bias_attrs, i)
            linear_weight_attr = get_attr(linear_weight_attrs, i)
            linear_bias_attr = get_attr(linear_bias_attrs, i)

            ffn_ln_scale_attr = get_attr(ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = get_attr(ffn_ln_bias_attrs, i)
            gate_weight_attr = get_attr(gate_weight_attrs, i)
            gate_bias_attr = get_attr(gate_bias_attrs, i)

            qkv_out_scales_attr = get_attr(qkv_out_scales_attrs, i)
            out_linear_out_scales_attr = get_attr(out_linear_out_scales_attrs, i)
            expert_weight1_out_scales_attr = get_attr(expert_weight1_out_scales_attrs, i)
            expert_weight2_out_scales_attr = get_attr(expert_weight2_out_scales_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
                dtype="float32",
            )
            ln_bias = self.create_parameter(
                attr=ln_bias_attr, shape=[embed_dim], is_bias=True, dtype="float32"
            )
            qkv_weight = self.create_parameter(
                shape=[3, num_heads, self.head_dim, embed_dim]
                if trans_qkvw
                else [embed_dim, 3, num_heads, self.head_dim],
                attr=qkv_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            qkv_bias = self.create_parameter(
                shape=[3, num_heads, self.head_dim],
                attr=qkv_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            linear_weight = self.create_parameter(
                shape=[num_heads * self.head_dim, embed_dim],
                attr=linear_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            linear_bias = self.create_parameter(
                shape=[embed_dim],
                attr=linear_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            qkv_out_scale = self.create_parameter(
                shape=[3 * embed_dim],
                attr=qkv_out_scales_attr,
                dtype="float32",
                is_bias=False
            )
            out_linear_out_scale = self.create_parameter(
                shape=[embed_dim],
                attr=out_linear_out_scales_attr,
                dtype="float32",
                is_bias=False
            )

            ffn_ln_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
                dtype="float32",
            )
            ffn_ln_bias = self.create_parameter(
                shape=[embed_dim], attr=ffn_ln_bias_attr, is_bias=True, dtype="float32"
            )
            gate_weight = self.create_parameter(
                shape=[d_model, num_expert * self.world_size],
                attr=gate_weight_attr,
                dtype=self._dtype,
                is_bias=False
            )
            gate_bias = self.create_parameter(
                shape=[num_expert * self.world_size],
                attr=gate_bias_attr,
                dtype=self._dtype,
                is_bias=True
            )

            # tensor model parallel
            if nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(qkv_bias)
                # row parallel
                _set_var_distributed(linear_weight)

            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            self.qkv_weights.append(qkv_weight)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_biases.append(linear_bias)
            self.qkv_out_scales.append(qkv_out_scale)
            self.out_linear_out_scales.append(out_linear_out_scale)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.gate_weights.append(gate_weight)
            self.gate_biases.append(gate_bias)

            for j in range(num_expert):
                expert_weight1_attr = get_attr(expert_weight1_attrs, i * num_expert + j)
                expert_bias1_attr = get_attr(expert_bias1_attrs, i * num_expert + j)
                expert_weight2_attr = get_attr(expert_weight2_attrs, i * num_expert + j)
                expert_bias2_attr = get_attr(expert_bias2_attrs, i * num_expert + j)

                expert_weight1 = self.create_parameter(
                    # shape=[d_model, dim_feedforward],
                    shape=[dim_feedforward, d_model],
                    attr=expert_weight1_attr,
                    dtype=self._dtype,
                    is_bias=False,
                    default_initializer=nn.initializer.KaimingUniform()
                )
                expert_bias1 = self.create_parameter(
                    shape=[dim_feedforward],
                    attr=expert_bias1_attr,
                    dtype=self._dtype,
                    is_bias=True,
                    default_initializer=nn.initializer.Constant(value=0.0)
                )
                expert_weight2 = self.create_parameter(
                    # shape=[dim_feedforward, d_model],
                    shape=[d_model, dim_feedforward],
                    attr=expert_weight2_attr,
                    dtype=self._dtype,
                    is_bias=False,
                    default_initializer=nn.initializer.KaimingUniform()
                )
                expert_bias2 = self.create_parameter(
                    shape=[d_model],
                    attr=expert_bias2_attr,
                    dtype=self._dtype,
                    is_bias=True,
                    default_initializer=nn.initializer.Constant(value=0.0)
                )
                expert_weight1_out_scale = self.create_parameter(
                    shape=[4 * embed_dim],
                    attr=expert_weight1_out_scales_attr,
                    dtype="float32",
                    is_bias=False
                )
                expert_weight2_out_scale = self.create_parameter(
                    shape=[embed_dim],
                    attr=expert_weight2_out_scales_attr,
                    dtype="float32",
                    is_bias=False
                )
                expert_weight1.name = "expert_" + expert_weight1.name
                expert_bias1.name = "expert_" + expert_bias1.name
                expert_weight2.name = "expert_" + expert_weight2.name
                expert_bias2.name = "expert_" + expert_bias2.name
                self.expert_weights1.append(expert_weight1)
                self.expert_biases1.append(expert_bias1)
                self.expert_weights2.append(expert_weight2)
                self.expert_biases2.append(expert_bias2)
                self.expert_weight1_out_scales.append(expert_weight1_out_scale)
                self.expert_weight2_out_scales.append(expert_weight2_out_scale)
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.name = name
        # int8
        self._int8_decorate()

    def forward(self, src, attn_mask=None, caches=None, seq_lens=None, beam_offset=None, time_step=None):
        """
        forward
        """
        cache_kv_out, final_out = _legacy_C_ops.fused_multi_transformer_moe_int8(
            src,
            list(self.ln_scales),
            list(self.ln_biases),
            list(self.qkv_weights),
            list(self.qkv_biases),
            caches,
            beam_offset,
            time_step,
            seq_lens,
            attn_mask,
            list(self.linear_weights),
            list(self.linear_biases),
            list(self.gate_weights),
            list(self.gate_biases),
            list(self.ffn_ln_scales),
            list(self.ffn_ln_biases),
            list(self.expert_weights1),
            list(self.expert_biases1),
            list(self.expert_weights2),
            list(self.expert_biases2),
            list(self.qkv_out_scales),
            list(self.out_linear_out_scales),
            list(self.expert_weight1_out_scales),
            list(self.expert_weight2_out_scales),
            caches,
            'pre_layer_norm',
            self.normalize_before,
            'epsilon',
            self._epsilon,
            'dropout_rate',
            self.dropout_rate,
            'is_test',
            not self.training,
            'dropout_implementation',
            'upscale_in_train',
            'act_method',
            self.activation,
            'trans_qkvw',
            self._trans_qkvw,
            'ring_id',
            self._ring_id,
            'topk',
            self.top_k,
            'mp_size',
            self.mp_size,
            'mp_rank',
            self.mp_rank,
            'num_expert',
            self.num_expert,
            'world_size',
            self.world_size,
            'moe_ring_id',
            -1 if self.group is None else self.group.id,
            'approximate',
            self.approximate,
            'qkv_in_scale',
            self.qkv_in_scale,
            'out_linear_in_scale',
            self.out_linear_in_scale,
            'expert_weight1_in_scale',
            self.expert_weight1_in_scale,
            'expert_weight2_in_scale',
            self.expert_weight2_in_scale
        )
        if caches is not None:
            return final_out, cache_kv_out
        return final_out

    def _int8_decorate(self, dtype="int8"):
        # tmp fix for INT8
        def trans_to_int8(l):
            for param in l:
                if param is not None:
                    with no_grad():
                        param_applied = _to_dtype(param, dtype)
        trans_to_int8(self.qkv_weights)
        trans_to_int8(self.linear_weights)
        trans_to_int8(self.expert_weights1)
        trans_to_int8(self.expert_weights2)
        self._dtype = "int8"


class FusedMultiTransformerMoeWeightOnly(Layer):
    """
    FusedMultiTransformerMoe
    """
    def __init__(
        self,
        d_model,
        embed_dim,
        num_heads,
        dim_feedforward,
        dropout_rate=0.0,
        activation="gelu",
        weight_dtype="int8",
        normalize_before=True,
        ln_scale_attrs=None,
        ln_bias_attrs=None,
        qkv_weight_attrs=None,
        qkv_scale_attrs=None,
        qkv_bias_attrs=None,
        linear_weight_attrs=None,
        linear_scale_attrs=None,
        linear_bias_attrs=None,
        gate_weight_attrs=None,
        gate_bias_attrs=None,
        ffn_ln_scale_attrs=None,
        ffn_ln_bias_attrs=None,
        expert_weight1_attrs=None,
        expert_scale1_attrs=None,
        expert_bias1_attrs=None,
        expert_weight2_attrs=None,
        expert_scale2_attrs=None,
        expert_bias2_attrs=None,
        epsilon=1e-5,
        num_layers=-1,
        nranks=1,
        ring_id=-1,
        num_expert=1,
        top_k=2,
        approximate=True,
        moe_group=None,
        mp_group=None,
        name=None,
    ):
        super(FusedMultiTransformerMoeWeightOnly, self).__init__()
        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            "but received {}".format(embed_dim)
        )
        assert (
            num_heads > 0
        ), "Expected nhead to be greater than 0, " "but received {}".format(
            num_heads
        )
        assert (
            dim_feedforward > 0
        ), "Expected dim_feedforward to be greater than 0, but received {}".format(
            dim_feedforward
        )
        # only support mp/dp
        # for moe config
        self.group = moe_group
        self.world_size = 1
        if self.group is not None:
            self.world_size = self.group.nranks
        self.num_expert = num_expert

        self.mp_rank = 0
        self.mp_size = 1
        if mp_group is not None and mp_group.nranks > 1:
            self.mp_rank = mp_group.rank
            self.mp_size = mp_group.nranks
        self.top_k = top_k
        self.approximate = approximate

        # origin fmt config
        self.normalize_before = normalize_before
        self._dtype = "float16"
        self._epsilon = epsilon
        self._ring_id = ring_id
        self._weight_dtype = weight_dtype

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
        assert num_heads % nranks == 0
        num_heads = num_heads // nranks

        if isinstance(qkv_weight_attrs, (list, tuple, ParameterList)):
            num_layers = len(qkv_weight_attrs)
        assert num_layers > 0

        self.ln_scales, self.ln_biases = ParameterList(), ParameterList()
        self.qkv_weights, self.qkv_scales, self.qkv_biases = ParameterList(), ParameterList(), ParameterList()
        self.linear_weights, self.linear_scales, self.linear_biases = ParameterList(), ParameterList(), ParameterList()
        self.gate_weights, self.gate_biases = ParameterList(), ParameterList()
        self.ffn_ln_scales, self.ffn_ln_biases = ParameterList(), ParameterList()
        self.expert_weights1, self.expert_scales1, self.expert_biases1 = ParameterList(), ParameterList(), ParameterList()
        self.expert_weights2, self.expert_scales2, self.expert_biases2 = ParameterList(), ParameterList(), ParameterList()
        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple, ParameterList)):
                assert len(attrs) == num_layers
                return attrs[idx]
            return attrs

        weight_int8 = False if self._weight_dtype == "int4" else True
        print(f"_weight_dtype: {self._weight_dtype}, weight_int8: {weight_int8}")
        for i in range(num_layers):
            ln_scale_attr = get_attr(ln_scale_attrs, i)
            ln_bias_attr = get_attr(ln_bias_attrs, i)
            qkv_weight_attr = get_attr(qkv_weight_attrs, i)
            qkv_scale_attr = get_attr(qkv_scale_attrs, i)
            qkv_bias_attr = get_attr(qkv_bias_attrs, i)
            linear_weight_attr = get_attr(linear_weight_attrs, i)
            linear_scale_attr = get_attr(linear_scale_attrs, i)
            linear_bias_attr = get_attr(linear_bias_attrs, i)

            ffn_ln_scale_attr = get_attr(ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = get_attr(ffn_ln_bias_attrs, i)
            gate_weight_attr = get_attr(gate_weight_attrs, i)
            gate_bias_attr = get_attr(gate_bias_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
                dtype="float32",
            )
            ln_bias = self.create_parameter(
                attr=ln_bias_attr, shape=[embed_dim], is_bias=True, dtype="float32"
            )
            qkv_weight = self.create_parameter(
                shape=[3,
                       num_heads,
                       self.head_dim if weight_int8 else int(self.head_dim / 2),
                       embed_dim],
                attr=qkv_weight_attr,
                dtype="uint8",
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0)
            )
            qkv_scale = self.create_parameter(
                shape=[int(3 * num_heads * self.head_dim)],
                attr=qkv_scale_attr,
                dtype=self._dtype,
                is_bias=False,
                default_initializer=Constant(1.0),
            )
            qkv_bias = self.create_parameter(
                shape=[3, num_heads, self.head_dim],
                attr=qkv_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            linear_weight = self.create_parameter(
                shape=[embed_dim if weight_int8 else int(embed_dim / 2),
                       int(num_heads * self.head_dim)],
                attr=linear_weight_attr,
                dtype="uint8",
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0)
            )
            linear_scale = self.create_parameter(
                shape=[embed_dim],
                attr=linear_scale_attr,
                dtype=self._dtype,
                is_bias=False,
                default_initializer=Constant(1.0),
            )
            linear_bias = self.create_parameter(
                shape=[embed_dim],
                attr=linear_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )

            ffn_ln_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
                dtype="float32",
            )
            ffn_ln_bias = self.create_parameter(
                shape=[embed_dim], attr=ffn_ln_bias_attr, is_bias=True, dtype="float32"
            )
            gate_weight = self.create_parameter(
                shape=[d_model, num_expert * self.world_size],
                attr=gate_weight_attr,
                dtype=self._dtype,
                is_bias=False
            )
            gate_bias = self.create_parameter(
                shape=[num_expert * self.world_size],
                attr=gate_bias_attr,
                dtype=self._dtype,
                is_bias=True
            )

            # tensor model parallel
            if nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(qkv_bias)
                # row parallel
                _set_var_distributed(linear_weight)

            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            self.qkv_weights.append(qkv_weight)
            self.qkv_scales.append(qkv_scale)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_scales.append(linear_scale)
            self.linear_biases.append(linear_bias)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.gate_weights.append(gate_weight)
            self.gate_biases.append(gate_bias)

            for j in range(num_expert):
                expert_weight1_attr = get_attr(expert_weight1_attrs, i * num_expert + j)
                expert_scale1_attr = get_attr(expert_scale1_attrs, i * num_expert + j)
                expert_bias1_attr = get_attr(expert_bias1_attrs, i * num_expert + j)
                expert_weight2_attr = get_attr(expert_weight2_attrs, i * num_expert + j)
                expert_scale2_attr = get_attr(expert_scale2_attrs, i * num_expert + j)
                expert_bias2_attr = get_attr(expert_bias2_attrs, i * num_expert + j)

                expert_weight1 = self.create_parameter(
                    shape=[dim_feedforward if weight_int8 else int(dim_feedforward / 2),
                           d_model],
                    attr=expert_weight1_attr,
                    dtype="uint8",
                    is_bias=False,
                    default_initializer=nn.initializer.Constant(value=0)
                )
                expert_scale1 = self.create_parameter(
                    shape=[dim_feedforward],
                    attr=expert_scale1_attr,
                    dtype=self._dtype,
                    is_bias=False,
                    default_initializer=nn.initializer.Constant(value=0.0)
                )
                expert_bias1 = self.create_parameter(
                    shape=[dim_feedforward],
                    attr=expert_bias1_attr,
                    dtype=self._dtype,
                    is_bias=True,
                    default_initializer=nn.initializer.Constant(value=0.0)
                )
                expert_weight2 = self.create_parameter(
                    shape=[d_model if weight_int8 else int(d_model / 2),
                           dim_feedforward],
                    attr=expert_weight2_attr,
                    dtype="uint8",
                    is_bias=False,
                    default_initializer=nn.initializer.Constant(value=0)
                )
                expert_scale2 = self.create_parameter(
                    shape=[d_model],
                    attr=expert_scale2_attr,
                    dtype=self._dtype,
                    is_bias=False,
                    default_initializer=nn.initializer.Constant(value=0.0)
                )
                expert_bias2 = self.create_parameter(
                    shape=[d_model],
                    attr=expert_bias2_attr,
                    dtype=self._dtype,
                    is_bias=True,
                    default_initializer=nn.initializer.Constant(value=0.0)
                )
                expert_weight1.name = "expert_" + expert_weight1.name
                expert_scale1.name = "expert_" + expert_scale1.name
                expert_bias1.name = "expert_" + expert_bias1.name
                expert_weight2.name = "expert_" + expert_weight2.name
                expert_scale2.name = "expert_" + expert_scale2.name
                expert_bias2.name = "expert_" + expert_bias2.name
                self.expert_weights1.append(expert_weight1)
                self.expert_scales1.append(expert_scale1)
                self.expert_biases1.append(expert_bias1)
                self.expert_weights2.append(expert_weight2)
                self.expert_scales2.append(expert_scale2)
                self.expert_biases2.append(expert_bias2)
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.name = name
        self._int8_decorate()
        self._share_expert_param(num_layers, num_expert, dim_feedforward, d_model, weight_int8)
        self._dtype = "int8"

    def forward(self, src, attn_mask=None, caches=None, seq_lens=None, beam_offset=None, time_step=None):
        """
        forward
        """
        cache_kv_out, final_out = _legacy_C_ops.fused_multi_transformer_moe_weight_only(
            src,
            list(self.ln_scales),
            list(self.ln_biases),
            list(self.qkv_weights),
            list(self.qkv_scales),
            list(self.qkv_biases),
            caches,
            beam_offset,
            time_step,
            seq_lens,
            attn_mask,
            list(self.linear_weights),
            list(self.linear_scales),
            list(self.linear_biases),
            list(self.gate_weights),
            list(self.gate_biases),
            list(self.ffn_ln_scales),
            list(self.ffn_ln_biases),
            list(self.expert_weights1),
            list(self.expert_scales1),
            list(self.expert_biases1),
            list(self.expert_weights2),
            list(self.expert_scales2),
            list(self.expert_biases2),
            caches,
            'pre_layer_norm',
            self.normalize_before,
            'epsilon',
            self._epsilon,
            'dropout_rate',
            self.dropout_rate,
            'is_test',
            not self.training,
            'dropout_implementation',
            'upscale_in_train',
            'act_method',
            self.activation,
            'weight_dtype',
            self._weight_dtype,
            'ring_id',
            self._ring_id,
            'topk',
            self.top_k,
            'mp_size',
            self.mp_size,
            'mp_rank',
            self.mp_rank,
            'num_expert',
            self.num_expert,
            'world_size',
            self.world_size,
            'moe_ring_id',
            -1 if self.group is None else self.group.id,
            'approximate',
            self.approximate
        )
        if caches is not None:
            return final_out, cache_kv_out
        return final_out

    def _int8_decorate(self):
        # tmp fix for INT8
        def trans_to_int8(l):
            for param in l:
                if param is not None:
                    with no_grad():
                        param_applied = _to_dtype(param, "int8")
        trans_to_int8(self.qkv_weights)
        trans_to_int8(self.linear_weights)
        trans_to_int8(self.expert_weights1)
        trans_to_int8(self.expert_weights2)
        
    def _share_expert_param(self, num_layers, num_expert, dim_feedforward, d_model, weight_int8):
        """
        share_param
        """
        def shard_tensor(dst_tensor, parent_tensor, pos):
            tmp = parent_tensor.value().get_tensor()._slice(pos, pos + 1)
            dst_tensor.value().get_tensor()._share_data_buffer(tmp, False)
            #print(dst_tensor)

        self.shared_weights1, self.shared_scales1, self.shared_biases1 = ParameterList(), ParameterList(), ParameterList()
        self.shared_weights2, self.shared_scales2, self.shared_biases2 = ParameterList(), ParameterList(), ParameterList()

        int8_dim_feedforward = dim_feedforward if weight_int8 else int(dim_feedforward / 2)
        int8_d_model = d_model if weight_int8 else int(d_model / 2)
        for i in range(num_layers):
            shared_weight1 = paddle.create_parameter(
                name=f"moe.expert.layer{i}.shared_weight1",
                shape=[num_expert, int8_dim_feedforward, d_model],
                dtype="uint8",
                default_initializer=nn.initializer.Constant(value=0)) 
            shared_scale1 = paddle.create_parameter(
                name=f"moe.expert.layer{i}.shared_scale1",
                shape=[num_expert, dim_feedforward],
                dtype=self._dtype,
                default_initializer=nn.initializer.Constant(value=0.0)) 
            shared_bias1 = paddle.create_parameter(
                name=f"moe.expert.layer{i}.shared_bias1",
                shape=[num_expert, dim_feedforward],
                dtype=self._dtype,
                default_initializer=nn.initializer.Constant(value=0.0)) 
            
            shared_weight2 = paddle.create_parameter(
                name=f"moe.expert.layer{i}.shared_weight2",
                shape=[num_expert, int8_d_model, dim_feedforward],
                dtype="uint8",
                default_initializer=nn.initializer.Constant(value=0)) 
            shared_scale2 = paddle.create_parameter(
                name=f"moe.expert.layer{i}.shared_scale2",
                shape=[num_expert, d_model],
                dtype=self._dtype,
                default_initializer=nn.initializer.Constant(value=0.0)) 
            shared_bias2 = paddle.create_parameter(
                name=f"moe.expert.layer{i}.shared_bias2",
                shape=[num_expert, d_model],
                dtype=self._dtype,
                default_initializer=nn.initializer.Constant(value=0.0)) 

            for j in range(self.num_expert):
                expert_idx = j + i * self.num_expert
                shard_tensor(self.expert_weights1[expert_idx], shared_weight1, j)
                shard_tensor(self.expert_scales1[expert_idx], shared_scale1, j)
                shard_tensor(self.expert_biases1[expert_idx], shared_bias1, j)
                shard_tensor(self.expert_weights2[expert_idx], shared_weight2, j)
                shard_tensor(self.expert_scales2[expert_idx], shared_scale2, j)
                shard_tensor(self.expert_biases2[expert_idx], shared_bias2, j)

            self.shared_weights1.append(shared_weight1)
            self.shared_scales1.append(shared_scale1) 
            self.shared_biases1.append(shared_bias1)

            self.shared_weights2.append(shared_weight2)
            self.shared_scales2.append(shared_scale2) 
            self.shared_biases2.append(shared_bias2)