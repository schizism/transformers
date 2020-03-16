
import numpy as np
import torch
import os
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Callable
import random

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)


prompt_text = 'can we please'
device = 'cpu'
length = 3


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
model.to(device)

input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)

from torch.quantization import QuantStub, DeQuantStub


def print_size_of_model(model, saveLocation):
    torch.save(model.state_dict(), saveLocation)
    tmp = 'Size (MB): ' + str(os.path.getsize(saveLocation)/1e6)
    os.remove(saveLocation)
    return tmp

print("model size before quantization: ", print_size_of_model(model, 'C:/Users/hanyu/internal_work/AML/distilgpt2-finetuned/tmp/temp.p'))


# setup configuration and prepare model
qconfig = torch.quantization.default_qconfig
print(qconfig)
torch.quantization.prepare(model, inplace=True)

# calibration
def evaluate(model, n_cases, tokenizer, fileLocation, device = 'cpu'):
    model.eval()
    cnt = 0
    ofile = open(fileLocation)
    with torch.no_grad():
        for text in ofile:
            tmp = text.strip()
            if not tmp:
                continue
            input_ids = tokenizer.encode(tmp, add_special_tokens=False, return_tensors="pt").to(device)
            output = model(input_ids)
            cnt += 1
            if cnt >= n_cases:
                ofile.close()
                return
    ofile.close()
    return

evaluate(model, n_cases = 100, tokenizer = tokenizer, fileLocation = 'C:/Users/hanyu/internal_work/AML/distilgpt2-finetuned/prediction_text.txt', device = 'cpu')

# conversion
torch.quantization.convert(model, inplace=True)

print("model size after quantization: ", print_size_of_model(model, 'C:/Users/hanyu/internal_work/AML/distilgpt2-finetuned/tmp/temp.p'))