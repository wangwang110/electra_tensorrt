from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import onnxruntime as ort
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from transformers import ElectraTokenizer, ElectraModel

ort_session = ort.InferenceSession('./electra.onnx')

plm_model_path = "/data_local/plm_models/electra_base"
model = ElectraModel.from_pretrained(plm_model_path)
tokenizer = ElectraTokenizer.from_pretrained(plm_model_path)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
input_ids = np.array(inputs["input_ids"])
token_type_ids = np.array(inputs["token_type_ids"])
attention_mask = np.array(inputs["attention_mask"])
# 获取数据，也可以自己构造

ort_inputs = {ort_session.get_inputs()[0].name: input_ids,
              ort_session.get_inputs()[1].name: token_type_ids,
              ort_session.get_inputs()[2].name: attention_mask
              }
outputs = ort_session.run(None, ort_inputs)[0]
print(outputs)
