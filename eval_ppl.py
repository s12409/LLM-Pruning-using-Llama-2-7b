import abc

from torch.xpu import device

import lm_eval
import argparse
import datetime
import torch
import os
import time
# import neutorch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig, LlamaTokenizer, pipeline, BitsAndBytesConfig
# from neutorch.conversion.linear import neu_linear
from lm_eval import evaluator
from lm_eval.base import BaseLM
from lm_eval import utils
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


class EvalLM(BaseLM):
    def __init__(
        self,
        model,
        tokenizer,
        # device="cuda:0",
        batch_size=1,
    ):
        super().__init__()

        # assert isinstance(device, str)
        assert isinstance(batch_size, int)

        # self._device = torch.device(device)
        self._device = model.device

        # self.model = model.to(self.device)
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # self.seqlen = 2048
        self.seqlen = 2048
        self.pipe = pipeline("text-generation", model=self.model, torch_dtype=torch.bfloat16, device_map="auto", tokenizer=self.tokenizer)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :50257]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
    # def _model_generate(self, context, max_length, eos_token_id):
    #     start_time = time.time()
    #     sequences = self.pipe(context,
    #                   do_sample=False,
    #                   top_k=10,
    #                   num_return_sequences=1,
    #                   eos_token_id=eos_token_id,
    #                   max_length=max_length)
    #     end_time = time.time()
    #     cpu_time = round(end_time - start_time, 2)
    #
    #     answer = sequences[0]['generated_text']
    #     tokens = len(self.tokenizer.tokenize(answer))
    #     #print("len of tokenizer tokenize:", tokens, "\n") # here is the tokenized length
    #     print("\n[", cpu_time, "s]", "[", round(tokens/cpu_time, 2), "tokens/s]")
    #     print(answer)
    #     print(self._device)
    #     return answer


def eval_ppl(lm, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // lm.seqlen

    # List to store negative log likelihoods
    nlls = []
    loss_lst = []
    print(f"nsamples: {nsamples}")

    # Loop through each batch
    # for i in tqdm(range(0, nsamples, bs)):
    #     # Calculate end index
    #     j = min(i + bs, nsamples)
    #
    #     # Prepare inputs and move to device
    #     inputs = testenc[:, (i * lm.seqlen):(j * lm.seqlen)].to(device)
    #     inputs = inputs.reshape(j - i, lm.seqlen)
    #     outputs = lm.model.model(inputs)
    #     hidden_states = outputs[0]
    #     lm_logits = lm.model.lm_head(hidden_states)
    #     # Shift logits and labels for next token prediction
    #     shift_logits = lm_logits[:, :-1, :].contiguous()
    #
    #     shift_labels = inputs[:, 1:].cuda()
    #
    #     # Compute loss
    #     loss_fct = nn.CrossEntropyLoss()
    #     loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    #
    #     # Calculate negative log likelihood
    #     neg_log_likelihood = loss.float() * lm.seqlen * (j - i)
    #
    #     # Append to list of negative log likelihoods
    #     nlls.append(neg_log_likelihood)
    #     # loss_lst.append(loss.float())
    for i in tqdm(range(0, nsamples, bs)):
        j = min(i + bs, nsamples)

        # Move input to device
        with torch.no_grad():
            inputs = testenc[:, (i * lm.seqlen):(j * lm.seqlen)].to(device)
            inputs = inputs.reshape(j - i, lm.seqlen)
            outputs = lm.model.model(inputs)
            hidden_states = outputs[0]
            lm_logits = lm.model.lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

            # Calculate negative log likelihood
            neg_log_likelihood = loss.float() * lm.seqlen * (j - i)
            nlls.append(neg_log_likelihood)
            loss_lst.append(loss.float())

        # Release memory
        del inputs, outputs, hidden_states, lm_logits, shift_logits
        torch.cuda.empty_cache()

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))

    return ppl.item(), torch.stack(nlls).sum() / (nsamples * lm.seqlen)


def main():
    batch_size = 1
    prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
    # test_model = "./pruned_model_pretrained/ratio-0.5_expand"
    # test_model = "./pruned_model_pretrained/ratio-0.25_expand"
    test_model = "../llama-2/7B"
    # model = LlamaForCausalLM.from_pretrained(test_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    device = "cuda:0"
    number_of_words = len(prompt.split())
    tokenizer = LlamaTokenizer.from_pretrained(test_model)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    model= LlamaForCausalLM.from_pretrained(
        # 'llama-2/7B',
        test_model,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map='auto',
        # low_cpu_mem_usage=True
    )
    # print("len of tokenizer tokenize:", len(tokenizer.tokenize(prompt))) # here is the tokenized length
    #
    # pruned model
    # model_load = torch.load('./LLM-Pruner/model-prune-0.25/model.pth', map_location='cpu')
    # pruned_model = model_load['model']
    #
    # tokenizer = model_load['tokenizer']
    #
    # # Specified your devices
    # device_ids = neutorch._C.get_available_devices()
    # print(device_ids)
    # neutorch._C.set_device(device_ids[:1])  # single card
    #
    # compiled_model_path = os.path.join(os.getcwd(), './LLM-Pruner/model-prune-0.25/data') if os.path.exists(os.path.join(os.getcwd(), './LLM-Pruner/model-prune-0.25/data')) else ''
    # print("Specified load model from", compiled_model_path)
    # model = neutorch.optimize(model, inplace=True, config_dir=compiled_model_path)
    # print(model.device)
    # p = pipeline("text-generation", model=model, torch_dtype=torch.bfloat16, device_map="auto", tokenizer=tokenizer)
    #
    lm = EvalLM(model, tokenizer, batch_size)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    ppl, _ = eval_ppl(lm, testenc, bs=1, device=device)
    print(f"perplexity: {ppl}")



if __name__ == '__main__':
    main()