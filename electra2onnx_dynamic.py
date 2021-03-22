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
input_ids = []
token_type_ids = []
attention_mask = []
for i in range(32):
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    input_ids.append(inputs["input_ids"])
    token_type_ids.append(inputs["token_type_ids"])
    attention_mask.append(inputs["attention_mask"])
input_ids = torch.cat(input_ids, 0)
token_type_ids = torch.cat(token_type_ids, 0)
attention_mask = torch.cat(attention_mask, 0)

# 将模型转成onnx过程中需要有输入，执行一遍前向推理
input_names = ["input_ids", "token_type_ids", "attention_mask"]
output_names = ["output_electra"]
# 输入的名称，只是为了可视化的时候，方便观看
ONNX_FILE_PATH = "./electra_dynamic.onnx"
torch.onnx.export(model,
                  (input_ids, token_type_ids, attention_mask),
                  ONNX_FILE_PATH, opset_version=10, verbose=True, input_names=input_names,
                  output_names=output_names,
                  dynamic_axes={
                      'input_ids': {
                          0: 'batch_size',
                          1: 'seq_len',
                      },
                      'token_type_ids': {
                          0: 'batch_size',
                          1: 'seq_len',
                      },
                      'attention_mask': {
                          0: 'batch_size',
                          1: 'seq_len',
                      },
                      'output_electra': {
                          0: 'batch_size',
                          1: 'seq_len',
                      }
                  },
                  export_params=True)

# 注意 dynamic_axes代表使用动态维度
# 如果默认参数，在用tensort推理时，只能使用和当前输入一样的维度，否则报错
