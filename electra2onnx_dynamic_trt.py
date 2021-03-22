from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import pycuda.driver as cuda

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import pycuda.autoinit
import tensorrt as trt
from transformers import ElectraTokenizer, ElectraModel
import numpy as np
# 需要在导入pycuda.driver后再导入pycuda.autoinit

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine


def setup_binding_shapes(engine: trt.ICudaEngine, context: trt.IExecutionContext, host_inputs, input_binding_idxs,
                         output_binding_idxs):
    # Explicitly set the dynamic input shapes, so the dynamic output
    # shapes can be computed internally
    for host_input, binding_index in zip(host_inputs, input_binding_idxs):
        context.set_binding_shape(binding_index, host_input.shape)
    assert context.all_binding_shapes_specified
    host_outputs = []
    device_outputs = []
    for binding_index in output_binding_idxs:
        output_shape = context.get_binding_shape(binding_index)
    # Allocate buffers to hold output results after copying back to host
    buffer = np.empty(output_shape, dtype=np.float32)
    host_outputs.append(buffer)
    # Allocate output buffers on device
    device_outputs.append(cuda.mem_alloc(buffer.nbytes))
    # 绑定输出shape
    utput_names = [engine.get_binding_name(binding_idx) for binding_idx in output_binding_idxs]
    return host_outputs, device_outputs


def get_binding_idxs(engine: trt.ICudaEngine, profile_index: int):
    """
    :param engine:
    :param profile_index:
    :return:
    """
    # Calculate start/end binding indices for current context's profile
    num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
    start_binding = profile_index * num_bindings_per_profile
    end_binding = start_binding + num_bindings_per_profile  # Separate input and output binding indices for convenience
    input_binding_idxs = []
    output_binding_idxs = []
    for binding_index in range(start_binding, end_binding):
        if engine.binding_is_input(binding_index):
            input_binding_idxs.append(binding_index)
        else:
            output_binding_idxs.append(binding_index)
    return input_binding_idxs, output_binding_idxs


def get_engine(onnx_file_path="", engine_file_path="", save_engine=True):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(
                network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 28  # 256M
            builder.max_batch_size = 1
            ###
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 28

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            # 设置每一个维度的最小输入，一常规输入和最大输入
            profile = builder.create_optimization_profile()
            profile.set_shape(network.get_input(0).name, (1, 5), (1, 5), (32, 8))
            config.add_optimization_profile(profile)
            profile.set_shape(network.get_input(1).name, (1, 5), (1, 5), (32, 8))
            config.add_optimization_profile(profile)
            profile.set_shape(network.get_input(2).name, (1, 5), (1, 5), (32, 8))
            config.add_optimization_profile(profile)

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            # engine = builder.build_cuda_engine(network, config)
            engine = builder.build_engine(network, config)
            print("Completed creating Engine")
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(save_engine)


# These two modes are dependent on hardwares
onnx_model_path = './electra_dynamic.onnx'
trt_engine_path = './electra_dynamic.trt'
# Build an engine
engine = get_engine(onnx_model_path, trt_engine_path)
# Create the context for this engine
# context = engine.create_execution_context()
# Allocate buffers for input and output
# inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host # bindings

# Do inference
plm_model_path = "/data_local/plm_models/electra_base"
model = ElectraModel.from_pretrained(plm_model_path)
tokenizer = ElectraTokenizer.from_pretrained(plm_model_path)

# inputs_data = tokenizer("Hello, my dog is cute", return_tensors="pt")
# input_ids = np.array(inputs_data["input_ids"], dtype=np.int32)
# token_type_ids = np.array(inputs_data["token_type_ids"], dtype=np.int32)
# attention_mask = np.array(inputs_data["attention_mask"], dtype=np.int32)


input_ids = []
token_type_ids = []
attention_mask = []
for i in range(10):
    inputs_data = tokenizer("Hello, my dog is cute", return_tensors="pt")
    input_ids.append(inputs_data["input_ids"].numpy())
    token_type_ids.append(inputs_data["token_type_ids"].numpy())
    attention_mask.append(inputs_data["attention_mask"].numpy())

input_ids = np.concatenate(input_ids, 0)
token_type_ids = np.concatenate(token_type_ids, 0)
attention_mask = np.concatenate(attention_mask, 0)

with engine.create_execution_context() as context:
    context.active_optimization_profile = 0
    # 获得输入输出的id
    input_binding_idxs, output_binding_idxs = get_binding_idxs(engine, context.active_optimization_profile)

    # Do inference
    host_inputs = [input_ids, token_type_ids, attention_mask]
    device_inputs = [cuda.mem_alloc(h_input.nbytes) for h_input in host_inputs]  # 为每个输入分配显存
    for h_input, d_input in zip(host_inputs, device_inputs):
        cuda.memcpy_htod(d_input, h_input)  # 将实际输入数据转移到相应显存区域
    host_outputs, device_outputs = setup_binding_shapes(engine, context, host_inputs, input_binding_idxs,
                                                        output_binding_idxs)
    # 计算输入shape并绑定，为输出分配显存
    bindings = device_inputs + device_outputs
    context.execute_v2(bindings)  # 运行
    for h_output, d_output in zip(host_outputs, device_outputs):
        cuda.memcpy_dtoh(h_output, d_output)  # 将输出从显存移至内存
    outputs = host_outputs[0]
    print(outputs)
