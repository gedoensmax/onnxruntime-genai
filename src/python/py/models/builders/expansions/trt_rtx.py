# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import onnx_ir as ir
import torch


class TRT_RTX:
    """
    TRT-RTX specific subgraph expansions
    """
    def make_layernorm_subgraph(self, name, **kwargs):
        # This method can be used to create multiple LayerNorm operations
        op_type = kwargs.pop("op_type")
        inputs = kwargs.pop("inputs")
        outputs = kwargs.pop("outputs")
        skip = kwargs.pop("skip")
        new_io_dtype = kwargs.pop("new_io_dtype")

        if op_type == "LayerNormalization":
            # Create LayerNorm op
            self.make_layernorm_op(name, op_type, inputs, outputs, skip, new_io_dtype, **kwargs)
    
        elif op_type == "SkipLayerNormalization":
            # Create subgraph to calculate SkipLayerNorm
            self.make_skip_layer_norm(
                name,
                root_input=inputs[0],
                skip_input=inputs[1],
                weight_name=inputs[2],
                bias_name=inputs[3],
                output_0=outputs[0],
                output_3=outputs[3] if len(outputs) > 3 else None,
                io_dtype=new_io_dtype,
                shape=["batch_size", "sequence_length", self.hidden_size],
            )

        elif op_type == "SimplifiedLayerNormalization":
            # Create subgraph to calculate RMSNorm
            self.make_simplified_layer_norm(
                name,
                root_input=inputs[0],
                weight_name=inputs[1],
                output_0=outputs[0],
                io_dtype=new_io_dtype,
                shape=["batch_size", "sequence_length", self.hidden_size],
            )

        elif op_type == "SkipSimplifiedLayerNormalization":
            # Create subgraph to calculate SkipRMSNorm
            self.make_skip_simplified_layer_norm(
                name,
                root_input=inputs[0],
                skip_input=inputs[1],
                weight_name=inputs[2],
                output_0=outputs[0],
                output_3=outputs[3] if len(outputs) > 3 else None,
                io_dtype=new_io_dtype,
                shape=["batch_size", "sequence_length", self.hidden_size],
            )

    def make_skip_simplified_layer_norm(
        self, basename, root_input, skip_input, weight_name, output_0, output_3, io_dtype, shape
    ):
        #                          root_input         skip_input
        #                              |                  |
        #                              +------------------+
        #                              |
        #                             Add-------------> output (1)
        #                              |
        #                      SimplifiedLayerNorm----> output (0)
        make_add_name = f"{basename}/Add"
        output_3 = f"{make_add_name}/output_0" if output_3 is None else output_3
        self.make_node("Add", inputs=[root_input, skip_input], outputs=[output_3], name=make_add_name)
        self.make_value(output_3, io_dtype, shape=["batch_size", "sequence_length", self.hidden_size])

        make_simplified_layer_norm_name = f"{basename}/skip_simplified_layer_norm"
        self.make_simplified_layer_norm(
            make_simplified_layer_norm_name, output_3, weight_name, output_0, io_dtype, shape=shape
        )

    def make_skip_layer_norm(
        self, basename, root_input, skip_input, weight_name, bias_name, output_0, output_3, io_dtype, shape
    ):
        #                          root_input         skip_input
        #                              |                  |
        #                              +------------------+
        #                              |
        #                             Add-------------> output (1)
        #                              |
        #                      LayerNormalization-----> output (0)
        make_add_name = f"{basename}/Add"
        output_3 = f"{make_add_name}/output_0" if output_3 is None else output_3
        self.make_node("Add", inputs=[root_input, skip_input], outputs=[output_3], name=make_add_name)
        self.make_value(output_3, io_dtype, shape=["batch_size", "sequence_length", self.hidden_size])

        make_layer_norm_name = f"{basename}/LayerNormalization"
        inputs = [output_3, weight_name, bias_name]

        kwargs = {"epsilon": self.layernorm_attrs["epsilon"]}
        kwargs.update({"axis": -1, "stash_type": 1})

        self.make_node("LayerNormalization", inputs=inputs, outputs=[output_0], name=make_layer_norm_name, **kwargs)
        self.make_value(output_0, io_dtype, shape=shape)

    # This expansion contrib-op can be updated / deprecated in the future.
    def make_simplified_layer_norm(self, basename, root_input, weight_name, output_0, io_dtype, shape):
        #                            Cast (float32) - most calc happens in higher precision
        #                              |
        #                      +-------+-------+
        #                      |               |
        #                     Pow              |
        #                      |               |
        #                  ReduceMean          |
        #                      |               |
        #                     Add              |
        #                      |               |
        #                    Sqrt              |
        #                      |               |
        #                     Div              |
        #                      |               |
        #                      +-------+-------+
        #                              |
        #                             Mul
        #                              |
        #                            Cast_1 (io_dtype - float16)
        #                              |
        #                            Mul_1

        make_cast_name = f"{basename}/Cast"
        self.make_cast(make_cast_name, root_input, ir.DataType.FLOAT, shape=shape)

        make_pow_name = f"{basename}/Pow"
        make_pow_inputs = [f"{make_cast_name}/output_0", "/model/constants/FLOAT/2"]

        self.make_node(
            "Pow", inputs=make_pow_inputs, outputs=[f"{make_pow_name}/output_0"], name=make_pow_name, domain=""
        )
        self.make_value(f"{make_pow_name}/output_0", ir.DataType.FLOAT, shape=shape)

        make_reducemean_name = f"{basename}/ReduceMean"
        make_reducemean_inputs = [f"{make_pow_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_reduce_mean(
            make_reducemean_name, make_reducemean_inputs, ir.DataType.FLOAT, keepdims=True, shape=shape
        )

        make_add_name = f"{basename}/Add"
        make_add_inputs = [
            f"{make_reducemean_name}/output_0",
            f"/model/constants/FLOAT/{self.layernorm_attrs['epsilon']}",
        ]
        self.make_add(make_add_name, make_add_inputs, ir.DataType.FLOAT, shape=shape)

        make_sqrt_name = f"{basename}/Sqrt"
        make_sqrt_inputs = [f"{make_add_name}/output_0"]
        self.make_sqrt(make_sqrt_name, make_sqrt_inputs, ir.DataType.FLOAT, shape=shape)

        make_div_name = f"{basename}/Div"
        make_div_inputs = ["/model/constants/FLOAT/1", f"{make_sqrt_name}/output_0"]
        self.make_div(make_div_name, make_div_inputs, ir.DataType.FLOAT, shape=shape)

        make_mul_name = f"{basename}/Mul"
        make_mul_inputs = [f"{make_div_name}/output_0", f"{make_cast_name}/output_0"]
        self.make_mul(make_mul_name, make_mul_inputs, ir.DataType.FLOAT, shape=shape)

        make_cast_1_name = f"{basename}/Cast_1"
        self.make_cast(make_cast_1_name, f"{make_mul_name}/output_0", dtype=io_dtype, shape=shape)

        make_mul_1_name = f"{basename}/Mul_1"
        make_mul_1_inputs = [f"{make_cast_1_name}/output_0", weight_name]

        self.make_node("Mul", inputs=make_mul_1_inputs, outputs=[output_0], name=make_mul_1_name)
        self.make_value(output_0, dtype=io_dtype, shape=shape)

    def make_padded_cache(self, small_cache, large_cache, pad_value=0.0):
        """Pad small cache to match large cache shape for uniform If node branches.

        This is used for TRT-RTX EP which requires uniform dimensions in both branches of If nodes.

        Args:
            small_cache: The smaller cache tensor to pad
            large_cache: The larger cache tensor (defines target shape)
            pad_value: Value to use for padding (1.0 for cos_cache, 0.0 for sin_cache)
        """
        target_shape = large_cache.shape
        if small_cache.shape == target_shape:
            return small_cache

        # Create padded tensor filled with pad_value
        padded_cache = torch.full(target_shape, pad_value, dtype=small_cache.dtype)
        # Copy original data to the beginning
        padded_cache[: small_cache.shape[0], :] = small_cache
        return padded_cache

    def make_split_if_nodes(
        self,
        basename,
        greater_name,
        cos_cache_name,
        sin_cache_name,
        cos_cache_large,
        sin_cache_large,
        cos_cache_small,
        sin_cache_small,
        cos_cache_large_name,
        sin_cache_large_name,
        cos_cache_small_name,
        sin_cache_small_name,
        small_cache_shape,
    ):
        """Create split If nodes for TRT-RTX to workaround trt-rtx multi-output bug.

        This is a TEMPORARY workaround for TRT-RTX bug where If nodes with
        multiple outputs

        Creates two separate If nodes instead of one:
        - {basename}/cos/If: Outputs cos_cache only
        - {basename}/sin/If: Outputs sin_cache only

        Both If nodes use the same condition and independently select their respective caches.
        """
        cos_if_name = f"{basename}/cos/If"

        cos_large_for_split = ir.node(
            "Constant",
            [],
            outputs=[
                ir.Value(
                    name=f"{cos_cache_large_name}_split",
                    type=ir.TensorType(self.io_dtype),
                    shape=ir.Shape(cos_cache_large.shape),
                )
            ],
            name="/large/cos_cache/Constant_split_cos",
            attributes=dict(value=ir.tensor(cos_cache_large)),
        )

        cos_small_for_split = ir.node(
            "Constant",
            [],
            outputs=[
                ir.Value(
                    name=f"{cos_cache_small_name}_split",
                    type=ir.TensorType(self.io_dtype),
                    shape=ir.Shape(small_cache_shape),
                )
            ],
            name="/small/cos_cache/Constant_split_cos",
            attributes=dict(value=ir.tensor(cos_cache_small)),
        )

        self.make_node(
            "If",
            inputs=[f"{greater_name}/output_0"],
            outputs=[cos_cache_name],
            name=cos_if_name,
            then_branch=ir.Graph(
                inputs=[],
                outputs=[cos_large_for_split.outputs[0]],
                nodes=[cos_large_for_split],
                name="large_cos_cache_graph",
            ),
            else_branch=ir.Graph(
                inputs=[],
                outputs=[cos_small_for_split.outputs[0]],
                nodes=[cos_small_for_split],
                name="small_cos_cache_graph",
            ),
        )

        # Create separate If node for sin_cache only
        sin_if_name = f"{basename}/sin/If"

        # Create unique constant nodes for sin to avoid tensor sharing
        sin_large_for_split = ir.node(
            "Constant",
            [],
            outputs=[
                ir.Value(
                    name=f"{sin_cache_large_name}_split",
                    type=ir.TensorType(self.io_dtype),
                    shape=ir.Shape(sin_cache_large.shape),
                )
            ],
            name="/large/sin_cache/Constant_split_sin",
            attributes=dict(value=ir.tensor(sin_cache_large)),
        )

        sin_small_for_split = ir.node(
            "Constant",
            [],
            outputs=[
                ir.Value(
                    name=f"{sin_cache_small_name}_split",
                    type=ir.TensorType(self.io_dtype),
                    shape=ir.Shape(small_cache_shape),
                )
            ],
            name="/small/sin_cache/Constant_split_sin",
            attributes=dict(value=ir.tensor(sin_cache_small)),
        )

        self.make_node(
            "If",
            inputs=[f"{greater_name}/output_0"],
            outputs=[sin_cache_name],
            name=sin_if_name,
            then_branch=ir.Graph(
                inputs=[],
                outputs=[sin_large_for_split.outputs[0]],
                nodes=[sin_large_for_split],
                name="large_sin_cache_graph",
            ),
            else_branch=ir.Graph(
                inputs=[],
                outputs=[sin_small_for_split.outputs[0]],
                nodes=[sin_small_for_split],
                name="small_sin_cache_graph",
            ),
        )

        # Create output values
        self.make_value(cos_cache_name, self.io_dtype, shape=["max_sequence_length", "head_dim / 2"])
        self.make_value(sin_cache_name, self.io_dtype, shape=["max_sequence_length", "head_dim / 2"])

    def make_whisper_attention_init(self):
        self.graph.opset_imports[""] = max(self.graph.opset_imports[""], 24)

    def make_whisper_encoder_attention(self, layer_id, attention, root_input, **kwargs):
        # Whisper encoder self-attention using the standard ONNX Attention op.
        # TensorRT-RTX requires Q, K, and V to have the same rank, so emit 4D
        # [batch, heads, sequence, head_size] tensors explicitly.
        basename = f"/model/layers.{layer_id}/attn"

        qkv_matmul_basename = f"{basename}/qkv_proj/MatMul"
        qkv_matmul_name = self.make_packed_matmul(
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            qkv_matmul_basename,
            root_input,
            seq_dim=self.max_source_positions,
        )
        qkv_path = f"{qkv_matmul_name}/output_0"

        q_bias = (
            attention.q_proj.bias
            if attention.q_proj.bias is not None
            else attention.q_proj.weight.new_zeros(attention.q_proj.weight.shape[0])
        )
        k_bias = (
            attention.k_proj.bias
            if attention.k_proj.bias is not None
            else attention.k_proj.weight.new_zeros(attention.k_proj.weight.shape[0])
        )
        v_bias = (
            attention.v_proj.bias
            if attention.v_proj.bias is not None
            else attention.v_proj.weight.new_zeros(attention.v_proj.weight.shape[0])
        )
        qkv_add_name = f"{basename}/qkv_proj/Add"
        self.make_packed_add(
            q_bias, k_bias, v_bias, qkv_add_name, root_input=qkv_path, seq_dim=self.max_source_positions
        )
        qkv_path = f"{qkv_add_name}/output_0"

        q_path = f"{basename}/qkv_proj/Split/output_0"
        k_path = f"{basename}/qkv_proj/Split/output_1"
        v_path = f"{basename}/qkv_proj/Split/output_2"
        self.make_split(
            f"{basename}/qkv_proj/Split",
            inputs=[qkv_path, f"/model/constants/INT64/[{self.q_size}, {self.kv_size}, {self.kv_size}]"],
            outputs=[q_path, k_path, v_path],
            dtypes=[self.io_dtype] * 3,
            shapes=[
                ["batch_size", self.max_source_positions, self.q_size],
                ["batch_size", self.max_source_positions, self.kv_size],
                ["batch_size", self.max_source_positions, self.kv_size],
            ],
            axis=-1,
        )

        q_heads = self.make_whisper_attention_heads(
            f"{basename}/q_proj", q_path, self.num_attn_heads, self.max_source_positions
        )
        k_heads = self.make_whisper_attention_heads(
            f"{basename}/k_proj", k_path, self.num_kv_heads, self.max_source_positions
        )
        v_heads = self.make_whisper_attention_heads(
            f"{basename}/v_proj", v_path, self.num_kv_heads, self.max_source_positions
        )

        attn_name = f"{basename}/Attention"
        attn_output = f"{attn_name}/output_0"
        self.make_node(
            "Attention",
            inputs=[q_heads, k_heads, v_heads],
            outputs=[attn_output],
            name=attn_name,
            is_causal=0,
            scale=self.attention_attrs["scale"],
        )
        self.make_value(
            attn_output,
            self.io_dtype,
            shape=["batch_size", self.num_attn_heads, self.max_source_positions, self.head_size],
        )

        attn_merged = self.make_whisper_attention_output(
            basename, attn_output, self.max_source_positions, self.max_source_positions
        )

        o_matmul_basename = f"{basename}/o_proj/MatMul"
        o_matmul_name = self.make_matmul(
            attention.out_proj,
            o_matmul_basename,
            attn_merged,
            seq_dim=self.max_source_positions,
        )

        o_add_name = f"{basename}/o_proj/Add"
        self.make_add_bias(
            attention.out_proj.bias,
            o_add_name,
            root_input=f"{o_matmul_name}/output_0",
            seq_dim=self.max_source_positions,
        )

        self.layernorm_attrs["skip_input"] = f"{o_add_name}/output_0"

    def make_whisper_decoder_attention(self, layer_id, attention, root_input, **kwargs):
        basename = f"/model/layers.{layer_id}/attn"
        past_k, past_v, present_k, present_v = self.make_key_value_cache_names(layer_id)

        qkv_matmul_basename = f"{basename}/qkv_proj/MatMul"
        qkv_matmul_name = self.make_packed_matmul(
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            qkv_matmul_basename,
            root_input,
        )
        qkv_path = f"{qkv_matmul_name}/output_0"

        q_bias = (
            attention.q_proj.bias
            if attention.q_proj.bias is not None
            else attention.q_proj.weight.new_zeros(attention.q_proj.weight.shape[0])
        )
        k_bias = (
            attention.k_proj.bias
            if attention.k_proj.bias is not None
            else attention.k_proj.weight.new_zeros(attention.k_proj.weight.shape[0])
        )
        v_bias = (
            attention.v_proj.bias
            if attention.v_proj.bias is not None
            else attention.v_proj.weight.new_zeros(attention.v_proj.weight.shape[0])
        )
        qkv_add_name = f"{basename}/qkv_proj/Add"
        self.make_packed_add(q_bias, k_bias, v_bias, qkv_add_name, root_input=qkv_path)
        qkv_path = f"{qkv_add_name}/output_0"

        q_path = f"{basename}/qkv_proj/Split/output_0"
        k_path = f"{basename}/qkv_proj/Split/output_1"
        v_path = f"{basename}/qkv_proj/Split/output_2"
        self.make_split(
            f"{basename}/qkv_proj/Split",
            inputs=[qkv_path, f"/model/constants/INT64/[{self.q_size}, {self.kv_size}, {self.kv_size}]"],
            outputs=[q_path, k_path, v_path],
            dtypes=[self.io_dtype] * 3,
            shapes=[
                ["batch_size", "sequence_length", self.q_size],
                ["batch_size", "sequence_length", self.kv_size],
                ["batch_size", "sequence_length", self.kv_size],
            ],
            axis=-1,
        )

        q_heads = self.make_whisper_attention_heads(f"{basename}/q_proj", q_path, self.num_attn_heads)
        k_heads = self.make_whisper_attention_heads(f"{basename}/k_proj", k_path, self.num_kv_heads)
        v_heads = self.make_whisper_attention_heads(f"{basename}/v_proj", v_path, self.num_kv_heads)

        self.make_concat(
            f"{basename}/k_cache/Concat",
            [past_k, k_heads],
            dtype=self.io_dtype,
            shape=["batch_size", self.num_kv_heads, "total_sequence_length", self.head_size],
            axis=2,
        )
        self.make_node(
            "Identity",
            inputs=[f"{basename}/k_cache/Concat/output_0"],
            outputs=[present_k],
            name=f"{basename}/k_cache/Identity",
        )
        self.make_value(
            present_k, self.io_dtype, shape=["batch_size", self.num_kv_heads, "total_sequence_length", self.head_size]
        )

        self.make_concat(
            f"{basename}/v_cache/Concat",
            [past_v, v_heads],
            dtype=self.io_dtype,
            shape=["batch_size", self.num_kv_heads, "total_sequence_length", self.head_size],
            axis=2,
        )
        self.make_node(
            "Identity",
            inputs=[f"{basename}/v_cache/Concat/output_0"],
            outputs=[present_v],
            name=f"{basename}/v_cache/Identity",
        )
        self.make_value(
            present_v, self.io_dtype, shape=["batch_size", self.num_kv_heads, "total_sequence_length", self.head_size]
        )

        attn_name = f"{basename}/Attention"
        attn_output = f"{attn_name}/output_0"
        self.make_node(
            "Attention",
            inputs=[q_heads, present_k, present_v],
            outputs=[attn_output],
            name=attn_name,
            is_causal=0,
            scale=self.attention_attrs["scale"],
        )
        self.make_value(
            attn_output,
            self.io_dtype,
            shape=["batch_size", self.num_attn_heads, "sequence_length", self.head_size],
        )

        attn_merged = self.make_whisper_attention_output(basename, attn_output, "sequence_length", "sequence_length")

        o_matmul_basename = f"{basename}/o_proj/MatMul"
        o_matmul_name = self.make_matmul(attention.out_proj, o_matmul_basename, attn_merged)

        o_add_name = f"{basename}/o_proj/Add"
        self.make_add_bias(attention.out_proj.bias, o_add_name, root_input=f"{o_matmul_name}/output_0")

        self.layernorm_attrs["skip_input"] = f"{o_add_name}/output_0"

    def make_whisper_decoder_cross_attention(self, layer_id, attention, root_input, **kwargs):
        basename = f"/model/layers.{layer_id}/cross_attn"

        q_matmul_basename = f"{basename}/q_proj/MatMul"
        q_matmul_name = self.make_matmul(attention.q_proj, q_matmul_basename, root_input)
        q_path = f"{q_matmul_name}/output_0"

        q_add_name = f"{basename}/q_proj/Add"
        self.make_add_bias(attention.q_proj.bias, q_add_name, root_input=q_path)
        q_path = f"{q_add_name}/output_0"

        q_heads = self.make_whisper_attention_heads(f"{basename}/q_proj", q_path, self.num_attn_heads)

        attn_name = f"{basename}/Attention"
        attn_output = f"{attn_name}/output_0"
        self.make_node(
            "Attention",
            inputs=[
                q_heads,
                self.input_names["past_key_cross"][layer_id],
                self.input_names["past_value_cross"][layer_id],
            ],
            outputs=[attn_output],
            name=attn_name,
            is_causal=0,
            scale=self.attention_attrs["scale"],
        )
        self.make_value(
            attn_output,
            self.io_dtype,
            shape=["batch_size", self.num_attn_heads, "sequence_length", self.head_size],
        )

        attn_merged = self.make_whisper_attention_output(basename, attn_output, "sequence_length", "sequence_length")

        o_matmul_basename = f"{basename}/o_proj/MatMul"
        o_matmul_name = self.make_matmul(attention.out_proj, o_matmul_basename, attn_merged)

        o_add_name = f"{basename}/o_proj/Add"
        self.make_add_bias(attention.out_proj.bias, o_add_name, root_input=f"{o_matmul_name}/output_0")

        self.layernorm_attrs["skip_input"] = f"{o_add_name}/output_0"

    def make_whisper_attention_heads(self, basename, tensor, heads, sequence_length="sequence_length"):
        reshape_name = f"{basename}/Reshape"
        self.make_reshape(
            reshape_name,
            [tensor, f"/model/constants/INT64/[0, 0, {heads}, {self.head_size}]"],
            dtype=self.io_dtype,
            shape=["batch_size", sequence_length, heads, self.head_size],
        )
        transpose_name = f"{basename}/Transpose"
        self.make_transpose(
            transpose_name,
            f"{reshape_name}/output_0",
            dtype=self.io_dtype,
            shape=["batch_size", heads, sequence_length, self.head_size],
            perm=[0, 2, 1, 3],
        )
        return f"{transpose_name}/output_0"

    def make_whisper_attention_output(self, basename, attn_output, input_sequence_length, output_sequence_length):
        attn_transpose_name = f"{basename}/Attention/Transpose"
        self.make_transpose(
            attn_transpose_name,
            attn_output,
            dtype=self.io_dtype,
            shape=["batch_size", input_sequence_length, self.num_attn_heads, self.head_size],
            perm=[0, 2, 1, 3],
        )

        attn_reshape_name = f"{basename}/Attention/Reshape"
        self.make_reshape(
            attn_reshape_name,
            [
                f"{attn_transpose_name}/output_0",
                f"/model/constants/INT64/[0, 0, {self.head_size * self.num_attn_heads}]",
            ],
            dtype=self.io_dtype,
            shape=["batch_size", output_sequence_length, self.head_size * self.num_attn_heads],
        )
        return f"{attn_reshape_name}/output_0"

