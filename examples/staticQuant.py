# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# following:
# https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Callable
import random

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)



def inferenceModel(model):
    def inferceModelSub(input_ids):
        return model(input_ids=input_ids)[0][:, -1, :]
    return inferceModelSub


def _generate_no_beam_search(
    inferenceModel: Callable,
    input_ids,
    cur_len,
    batch_size,
    eos_token_ids,
    max_length = 20,
    do_sample = True,
    temperature = 1.0,
    top_k = 0,
    top_p = 0.9,
    repetition_penalty = 1.0,
    pad_token_id = 0,
    device = "cpu"
):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """
    # current position / max lengths / length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    while cur_len < max_length:
        next_token_logits = inferenceModel(input_ids = input_ids)
        # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
        # if repetition_penalty != 1.0:
        #     for i in range(batch_size):
        #         for previous_token in set(input_ids[i].tolist()):
        #             # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
        #             if next_token_logits[i, previous_token] < 0:
        #                 next_token_logits[i, previous_token] *= repetition_penalty
        #             else:
        #                 next_token_logits[i, previous_token] /= repetition_penalty
        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            # Top-p/top-k filtering
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            # Sample
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)
        # update generations and finished sentences
        tokens_to_add = next_token * unfinished_sents + pad_token_id * (1 - unfinished_sents)
        tokens_to_add = tokens_to_add.to(device)
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        for eos_token_id in eos_token_ids:
            unfinished_sents.mul_(tokens_to_add.ne(eos_token_id).long())
        cur_len = cur_len + 1
        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break
    # add eos_token_ids to unfinished sentences
    if cur_len == max_length:
        input_ids = torch.cat((input_ids, torch.empty(batch_size, 1, dtype=torch.long, device = device).fill_(eos_token_ids[0])), dim = 1)
        # input_ids[:, -1].masked_fill_(unfinished_sents.to(dtype=torch.bool), eos_token_ids[0])
    return input_ids




def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def pad_front(sequences, padding_value=0):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, max_len - length:, ...] = tensor
    return out_tensor


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

model.to(device)

input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
output_sequences = _generate_no_beam_search(
    inferenceModel = inferenceModel(model),
    input_ids = input_ids,
    cur_len = input_ids.shape[1],
    batch_size = input_ids.shape[0],
    eos_token_ids= [tokenizer.eos_token_id],
    max_length = length + input_ids.shape[1],
    device = device)

text = []
for i, tensor in enumerate(output_sequences):
    tmp = tokenizer.decode(tensor.tolist(), clean_up_tokenization_spaces=True)
    text.append(tmp)
    print(tmp)









# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# scratch
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
#
#
# def run_quant_fun(model, input_ids):
#     return model(input_ids)
#
# quant_model = torch.quantization.quantize(model, run_quant_fun, input_ids, mapping=None, inplace=False)
#
#
#
#
