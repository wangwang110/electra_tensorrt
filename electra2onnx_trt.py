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

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(onnx_file_path="", engine_file_path="", save_engine=True):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(
                network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 28  # 256M
            # builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
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


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


# These two modes are dependent on hardwares
onnx_model_path = './electra.onnx'
trt_engine_path = './electra.trt'
# Build an engine
engine = get_engine(onnx_model_path, trt_engine_path)
# Create the context for this engine
context = engine.create_execution_context()
# Allocate buffers for input and output
inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host # bindings

# Do inference
plm_model_path = "/data_local/plm_models/electra_base"
model = ElectraModel.from_pretrained(plm_model_path)
tokenizer = ElectraTokenizer.from_pretrained(plm_model_path)
inputs_data = tokenizer("Hello, my dog is cute", return_tensors="pt")
inputs[0].host = np.array(inputs_data["input_ids"], dtype=np.int32)
inputs[1].host = np.array(inputs_data["token_type_ids"], dtype=np.int32)
inputs[2].host = np.array(inputs_data["attention_mask"], dtype=np.int32)

trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)  # numpy data
print(trt_outputs[0])
