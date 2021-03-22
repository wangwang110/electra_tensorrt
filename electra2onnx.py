from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import ElectraTokenizer, ElectraModel

plm_model_path = "/data_local/plm_models/electra_base"
model = ElectraModel.from_pretrained(plm_model_path)
tokenizer = ElectraTokenizer.from_pretrained(plm_model_path)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# 将模型转成onnx过程中需要有输入，执行一遍前向推理
input_names = ["input_ids", "token_type_ids", "attention_mask"]
output_names = ["output_electra"]
# 输入的名称，只是为了可视化的时候，方便观看
ONNX_FILE_PATH = "./electra.onnx"
torch.onnx.export(model,
                  (inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]),
                  ONNX_FILE_PATH, opset_version=10, verbose=True, input_names=input_names,
                  output_names=output_names,
                  export_params=True)
