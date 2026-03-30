import abc
import lm_eval
import argparse
import datetime
import torch
import os
import time
# import neutorch
import torch.nn.functional as F
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

def hash_args(attr, args):
    dat = json.dumps([attr] + list(args))
    return hashlib.sha256(dat.encode("utf-8")).hexdigest()


class CacheHook:
    def __init__(self, cachinglm) -> None:
        if cachinglm is None:
            self.dbdict = None
            return

        self.dbdict = cachinglm.dbdict

    def add_partial(self, attr, req, res) -> None:
        if self.dbdict is None:
            return
        hsh = hash_args(attr, req)
        self.dbdict[hsh] = res


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

        self.seqlen = 2048

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



def main():
    batch_size = 16
    # task = "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
    # task = "hendrycksTest-abstract_algebra,"
    # task = "openbookqa"
    task = "arc_challenge"
    # task = "hellaswag"
    # ##################################
    # test_model = "./pruned_model_pretrained/ratio-0.5_expand"
    # test_model = "./pruned_model_pretrained/ratio-0.25_expand"
    test_model = "../llama-2/7B"
    # model = LlamaForCausalLM.from_pretrained(test_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    tokenizer = LlamaTokenizer.from_pretrained(test_model)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    model = LlamaForCausalLM.from_pretrained(
        # 'llama-2/7B',
        test_model,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map='auto',
        # low_cpu_mem_usage=True
    )

    # print(type(model))
    # task_manager = lm_eval.tasks.TaskManager()

    lm = EvalLM(model, tokenizer, batch_size)

    results = evaluator.simple_evaluate(  # call simple_evaluate
        model=lm,
        tasks=task.split(","),
        num_fewshot=0,
        batch_size=batch_size,
        limit=None,
        no_cache=True,
    )

    t_results = results["results"]
    acc_list = [
        t_results[key]["acc"] for key in t_results.keys() if "acc" in t_results[key]
    ]
    t_results["mean"] = sum(acc_list) / len(acc_list)
    results.update(t_results)
    print(results)
    # print mean
    print(f"\n\n===== mean acc: {sum(acc_list) / len(acc_list)} =====\n\n")

    pass


if __name__ == '__main__':
    main()
