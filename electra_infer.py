# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import ElectraTokenizer, ElectraModel

plm_model_path = "/data_local/plm_models/electra_base"
model = ElectraModel.from_pretrained(plm_model_path)
tokenizer = ElectraTokenizer.from_pretrained(plm_model_path)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
print(outputs)
