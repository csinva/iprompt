import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from copy import deepcopy
checkpoint = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, output_hidden_states=True)


# prepare inputs
raw_inputs = ["1+3=4"]
inputs = tokenizer(raw_inputs, return_tensors="pt")
print(inputs.keys())

# predict
outputs = model(**inputs)
