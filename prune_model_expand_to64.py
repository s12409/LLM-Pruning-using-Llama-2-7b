import abc
# import lm_eval
import argparse
import datetime
import torch
import os
import time
# import neutorch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig, LlamaTokenizer, pipeline
# from neutorch.conversion.linear import neu_linear
# from lm_eval import evaluator
# from lm_eval.base import BaseLM
# from lm_eval import utils
# from lm_eval.api.model import LM
from tqdm import tqdm
import hashlib
import json
import transformers
from datasets import load_dataset
import LLMPruner.torch_pruning as tp
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts
import logging


def model_expand(model):
    dim_list = [11008, 10048, 9024, 8000, 7040, 6016]
    # dim_list = [11008, 9024, 7040]
    for i in tqdm(range(4, 30)):
        features_num = model.model.layers[i].mlp.gate_proj.weight.data.size(0)
        # expand_num = 11008
        # if features_num % 64 == 0:
        #     expand_num = features_num
        # else:
        expand_num = features_num if features_num % 64 == 0 else (features_num // 64 + 1) * 64
        # for j in dim_list:
        #     if features_num <= j:
        #         expand_num = j
        #     else:
        #         break
        print(expand_num)

        model = gate_proj_expand(model, i, expand_num)
        model = down_proj_expand(model, i, expand_num)
        model = up_proj_expand(model, i, expand_num)

    return model

def gate_proj_expand(model, layer, expand_features):

    old_weight = model.model.layers[layer].mlp.gate_proj.weight.data
    print(old_weight.size(0), old_weight.size(1))
    print(model.model.layers[layer].mlp.gate_proj.weight.dtype)
    zeros = torch.zeros(expand_features - old_weight.size(0), old_weight.size(1), dtype=torch.bfloat16)
    print(zeros.shape)
    new_weight = torch.cat((old_weight, zeros), dim=0)
    new_linear_layer = nn.Linear(in_features=4096, out_features=expand_features, bias=False)
    new_linear_layer.weight.data = new_weight

    model.model.layers[layer].mlp.gate_proj = new_linear_layer
    print(model.model.layers[layer].mlp.gate_proj.weight.shape)
    return model

def down_proj_expand(model, layer, expand_features):

    old_weight = model.model.layers[layer].mlp.down_proj.weight.data
    print(old_weight.size(0), old_weight.size(1))
    print(model.model.layers[layer].mlp.down_proj.weight.shape)
    zeros = torch.zeros(old_weight.size(0), expand_features - old_weight.size(1), dtype=torch.bfloat16)
    print(zeros.shape)
    new_weight = torch.cat((old_weight, zeros), dim=1)
    new_linear_layer = nn.Linear(in_features=expand_features, out_features=4096, bias=False)
    new_linear_layer.weight.data = new_weight

    model.model.layers[layer].mlp.down_proj = new_linear_layer
    print(model.model.layers[layer].mlp.down_proj.weight.shape)
    return model

def up_proj_expand(model, layer, expand_features):

    old_weight = model.model.layers[layer].mlp.up_proj.weight.data
    print(old_weight.size(0), old_weight.size(1))
    print(model.model.layers[layer].mlp.up_proj.weight.shape)
    zeros = torch.zeros(expand_features - old_weight.size(0), old_weight.size(1), dtype=torch.bfloat16)
    print(zeros.shape)
    new_weight = torch.cat((old_weight, zeros), dim=0)
    new_linear_layer = nn.Linear(in_features=4096, out_features=expand_features, bias=False)
    new_linear_layer.weight.data = new_weight

    model.model.layers[layer].mlp.up_proj = new_linear_layer
    print(model.model.layers[layer].mlp.up_proj.weight.shape)
    return model

# def neuchips_compile(model,folder_path):
#     # Specified your devices
#     device_ids = neutorch._C.get_available_devices()
#     print(device_ids)
#     neutorch._C.set_device(device_ids[:1])  # single card
#
#     compiled_model_path = os.path.join(os.getcwd(), folder_path) if os.path.exists(os.path.join(os.getcwd(), folder_path)) else ''
#     print("Specified load model from", compiled_model_path)
#     model = neutorch.optimize(model, inplace=True, config_dir=compiled_model_path)
#     return model


def main():
    batch_size = 16
    prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
    test_model = "./llm_pruner/pruned_model_pretrained/ratio-0.5_not_expand"
    expand_to_64_model = "llm_pruner/pruned_model_pretrained/ratio-0.5_expand"
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # log setting
    logging.basicConfig(filename=f'{expand_to_64_model}/time_llama_to_64.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    ##################

    # pruned model
    # model_load = torch.load('./LLM-Pruner/model-prune-0.25/model.pth', map_location='cpu')
    # pruned_model = model_load['model']
    # pruned_tokenizer = model_load['tokenizer']
    pruned_model = LlamaForCausalLM.from_pretrained(test_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    pruned_tokenizer = LlamaTokenizer.from_pretrained(test_model)

    # ##################################




    testenc = pruned_tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    # print(pruned_model)
    print("pruned model")

    model = model_expand(pruned_model)
    print(model)
    model.save_pretrained(expand_to_64_model)

    pruned_tokenizer.save_pretrained(expand_to_64_model)

    # partial_model = list(model.model.named_children())
    #
    # parameters = []
    # for partial in partial_model[1][1]:
    #     parameter = sum([p.numel() for p in partial.parameters() if p.requires_grad])
    #     parameters.append(parameter)
    # print(parameters)


if __name__ == '__main__':
    main()