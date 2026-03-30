#!/usr/bin/env python3


import argparse
import datetime
import torch
import os
import time


from transformers import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaConfig,
    LlamaTokenizer,
    pipeline,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification
)
from peft import PeftModel
import os
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP

import LLMPruner.torch_pruning as tp
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts




class LlamaTestbed:

    def __init__(self):
        pass

    def run(self, test_model, prompt, max_tokens, verify_mode, verify_answer):
        print("[Run Testbed] run - ", datetime.datetime.now())
        answer = self.__test_model_logits(test_model, prompt, max_tokens, verify_mode)
        if verify_mode:
            assert(answer == verify_answer)
        print("[Run Testbed] end - ", datetime.datetime.now())

    def __inference(self, p, tokenizer, prompt, max_tokens):
        start_time = time.time()
        sequences = p(prompt,
                      do_sample=False,
                      top_k=10,
                      num_return_sequences=1,
                      eos_token_id=tokenizer.eos_token_id,
                      max_length=max_tokens)
        end_time = time.time()
        cpu_time = round(end_time - start_time, 2)

        answer = sequences[0]['generated_text']
        tokens = len(tokenizer.tokenize(answer))
        #print("len of tokenizer tokenize:", tokens, "\n") # here is the tokenized length
        print("\n[", cpu_time, "s]", "[", round(tokens/cpu_time, 2), "tokens/s]")
        print(answer)
        return answer

    def __test_model_logits(self, test_model, prompt, max_tokens, verify_mode):
        # m = LlamaForCausalLM.from_pretrained(test_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        # m = torch.load('./llm_pruner/model.pth', map_location='cpu')['model']
        # m = AutoModelForCausalLM.from_pretrained(
        #     pretrained_model_name_or_path='Models/qlora-10000-Merged',
        #     # load_in_4bit=True,
        #     # device_map='auto',
        #     # max_memory=80000,
        #     # torch_dtype=torch.float32,
        #     # quantization_config=BitsAndBytesConfig(
        #     #     load_in_4bit=True,
        #     #     llm_int8_threshold= 6.0,
        #     #     llm_int8_enable_fp32_cpu_offload= False,
        #     #     bnb_4bit_compute_dtype=torch.float32,
        #     #     bnb_4bit_use_double_quant=True,
        #     #     bnb_4bit_quant_type='nf4'
        #     # ),
        # m = LlamaForCausalLM.from_pretrained(
        #     'llama-2/7B',
        #     load_in_4bit=True,
        #     device_map='auto',
        #     # max_memory=80000,
        #     # torch_dtype=torch.int8,
        #     quantization_config=BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         # bnb_4bit_compute_dtype=torch.bfloat16,
        #         bnb_4bit_use_double_quant=True,
        #         bnb_4bit_quant_type='nf4'
        #     ),
        # )
        quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
        )
        m = AutoModelForCausalLM.from_pretrained(
            'llama-2/7B',
            # "llm_pruner/pruned_model_pretrained/ratio-0.5_expand",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map='auto',
            # low_cpu_mem_usage=True
        )
        base_model_id = "llama-2/7B"
        lora_model = "qlora/output/checkpoint-10000"
        # m = LlamaForCausalLM.from_pretrained(base_model_id)
        # m = PeftModel.from_pretrained(m, lora_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # m.to(device)
        print(f"Model is in dtype: {next(m.parameters()).dtype}")
        for name, param in m.named_parameters():
            if hasattr(param, 'quant_state'):
                print(f"{name} is quantized: {param.quant_state}")
        number_of_words = len(prompt.split())
        tokenizer = LlamaTokenizer.from_pretrained(test_model)
        #print("len of tokenizer tokenize:", len(tokenizer.tokenize(prompt))) # here is the tokenized length


        p = pipeline("text-generation", model=m, torch_dtype=torch.bfloat16, device_map="auto", tokenizer=tokenizer)
        print(m.device)
        print("Model dtype:", m.dtype)
        print(p)
        print("\n\nInference ************************************************")
        answer = self.__inference(p, tokenizer, prompt, max_tokens)
        if verify_mode:
            return answer

        while True:
            prompt = input("Enter your prompt string (or 'exit' to stop): ")
            if not prompt:
                print("You didn't enter anything. Please try again.")
                continue  # Continue the loop to prompt for input again
            elif prompt.lower() == 'exit':
                break  # Exit the loop if the user enters 'exit'

            max_tokens_str = input("Enter your max tokens number (or 'exit' to stop): ")
            if not max_tokens_str:
                print("You didn't enter anything. Use previous max tokens number ", max_tokens)
            elif max_tokens_str.lower() == 'exit':
                break  # Exit the loop if the user enters 'exit'
            else:
                try:
                    max_tokens = int(max_tokens_str)
                except ValueError:
                    print("You didn't enter anything. Use privious max tokens number ", max_tokens)

            answer = self.__inference(p, tokenizer, prompt, max_tokens)

        return answer


class ArgParser():

    def __init__(self):
        self.default_test_model = "./llama-2/7B"
        # self.default_test_model = "./llm_pruner"
        #self.default_test_model = "./data/sub0"
        # self.default_test_model = "/data/models/meta-llama/Llama-2-7b-hf"
        # self.default_test_model = "/data/models/meta-llama/Llama-2-13b-chat-hf"
        self.default_prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
        self.default_max_tokens = 50
        self.default_verify_mode = False
        self.default_verify_answer = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. She wanted to be a part of something bigger than herself. She wanted to be a part"

    def get_user_parameters(self):
        arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        arg_parser.add_argument('--test_model', type=str, help=f'Test model (default is \'{self.default_test_model}\')', default=self.default_test_model)
        arg_parser.add_argument('--prompt', type=str, help=f'Prompt string (default is \'{self.default_prompt}\')', default=self.default_prompt)
        arg_parser.add_argument('--max_tokens', type=int,  help=f'Max tokens number (default is {self.default_max_tokens})', default=self.default_max_tokens)
        arg_parser.add_argument('--verify_mode', type=bool,  help=f'Verify mode only for debug (default is {self.default_verify_mode})', default=self.default_verify_mode)
        arg_parser.add_argument('--verify_answer', type=str,  help=f'Verify answer when verify mode is true (default is {self.default_verify_answer})', default=self.default_verify_answer)

        args = arg_parser.parse_args()
        test_model = args.test_model
        prompt = args.prompt
        max_tokens = args.max_tokens
        verify_mode = args.verify_mode
        verify_answer = args.verify_answer

        print("test_model :", test_model)
        print("prompt :", prompt)
        print("max_tokens :", max_tokens)
        if verify_mode:
            print("verify_mode :", verify_mode)
            print("verify_answer :", verify_answer)
        print("")

        return [test_model, prompt, max_tokens, verify_mode, verify_answer]


def main():
    arg_parser = ArgParser()
    [test_model, prompt,  max_tokens, verify_mode, verify_answer] = arg_parser.get_user_parameters()

    testbed = LlamaTestbed()
    testbed.run(test_model, prompt, max_tokens, verify_mode, verify_answer)


if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()




