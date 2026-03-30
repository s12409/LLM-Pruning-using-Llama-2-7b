import abc
# import lm_eval
import argparse
import datetime
import torch
import os
import time
# import neutorch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig, LlamaTokenizer, pipeline, AutoModelForCausalLM, \
    AutoTokenizer
# from neutorch.conversion.linear import neu_linear
# from lm_eval import evaluator
# from lm_eval.base import BaseLM
# from lm_eval import utils
# from lm_eval.api.model import LM
from tqdm import tqdm
import hashlib
import json
import transformers
import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple
from torch import nn
import torch
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP

import LLMPruner.torch_pruning as tp
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts
import logging
from datasets import load_dataset
import tempfile




def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main():
    # parameters
    batch_size = 16
    test_model = "./llama-2/7B"
    set_random_seed(0)
    pruner_type = "taylor"
    taylor = 'param_first'
    block_attention_layer_start, block_attention_layer_end = 4, 30
    block_mlp_layer_start, block_mlp_layer_end = 4, 30
    iterative_steps = 1
    pruning_ratio = 0.5
    num_examples = 10
    save_model = True
    best_checkpoint_path = f'./pruned_model_pretrained/ratio-0.5_not_expand'
    ##################
    if not os.path.exists(best_checkpoint_path):
        os.makedirs(best_checkpoint_path)

    # log setting
    logging.basicConfig(filename=f'{best_checkpoint_path}/pruned_model_pretrained-{pruning_ratio}.log', filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    ##################

    model = AutoModelForCausalLM.from_pretrained(test_model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(test_model)

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    forward_prompts = torch.tensor([
        [1, 306, 4658, 278, 6593, 310, 2834, 338],
        [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
    ])
    # block_wise pruning
    imp = llama_pruner.TaylorImportance(group_reduction='sum', taylor=taylor)
    kwargs = {
        "importance": imp,
        "global_pruning": 'store_true',
        "iterative_steps": iterative_steps,
        "ch_sparsity": pruning_ratio,
        "ignored_layers": [],
        "channel_groups": {
        },
        "consecutive_groups": {
            layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
        },
        "customized_pruners": {
            LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
        },
        "root_module_types": None,
        "root_instances": [model.model.layers[i].self_attn.q_proj for i in
                           range(block_attention_layer_start, block_attention_layer_end)] +
                          [model.model.layers[i].mlp.gate_proj for i in
                           range(block_mlp_layer_start, block_mlp_layer_end)]
    }
    logging.info(
        "Pruning Attention Layer = {}".format(list(range(block_attention_layer_start, block_attention_layer_end))))
    logging.info("Pruning MLP Layer = {}".format(list(range(block_mlp_layer_start, block_mlp_layer_end))))

    pruner = tp.pruner.MetaPruner(
        model,
        forward_prompts,
        **kwargs
    )
    model.zero_grad()

    logging.info("Start Pruning")
    for i in range(iterative_steps):
        if pruner_type in ['taylor']:
            example_prompts = get_examples('bookcorpus', tokenizer, num_examples, seq_len=64).to('cpu')
            logging.info("Start Backwarding in iterative steps = {}...".format(i))
            if taylor in ['param_mix', 'param_second']:
                for j in range(num_examples):
                    batch_input = example_prompts[j].unsqueeze(0)
                    loss = model(batch_input, labels=batch_input).loss
                    logging.info("Loss = {}".format(loss))
                    loss.backward()

                    for module_param in model.parameters():
                        module_param.grad = module_param.grad * module_param.grad / num_examples
                        if hasattr(module_param, 'acc_grad'):
                            module_param.acc_grad += module_param.grad
                        else:
                            module_param.acc_grad = copy.deepcopy(module_param.grad)
                    model.zero_grad()
                    del loss.grad

            loss = model(example_prompts, labels=example_prompts).loss
            logging.info("Loss = {}".format(loss))
            loss.backward()

        pruner.step()

        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(
            "After Iter {}/{}, #parameters: {}".format(i + 1, iterative_steps, after_pruning_parameters))

        # modify inferece-related attributes
        for layer in model.model.layers:
            layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

    # Clean the gradient in the model
    model.zero_grad()
    for name, module in model.named_parameters():
        if 'weight' in name:
            module.grad = None

    del pruner
    logging.info("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters,
                                                                               after_pruning_parameters,
                                                                               100.0 * after_pruning_parameters / before_pruning_parameters))

    if not os.path.isdir(best_checkpoint_path):
        os.makedirs(best_checkpoint_path)

    gc.collect()
    if save_model:

        model.save_pretrained(best_checkpoint_path)

        tokenizer.save_pretrained(best_checkpoint_path)

    pass


if __name__ == '__main__':
    main()
