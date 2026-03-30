import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
import string




def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('\n', ' ')
    return text


class TextTokenized(Dataset):
    def __init__(self, articles, highlights, tokenizer, max_len=512):
        self.articles = articles
        self.highlights = highlights
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, index):
        article = self.articles[index]
        highlight = self.highlights[index]

        input_encoding = self.tokenizer(
            article, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            highlight, max_length=128, truncation=True, padding="max_length", return_tensors="pt"
        )

        labels = target_encoding["input_ids"].squeeze()
        decoder_input_ids = torch.full_like(labels, fill_value=-100)  # Defaulting to -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }



def dataset_load(data_path, tokenizer):


    train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
    val_data = pd.read_csv(os.path.join(data_path, 'validation.csv'))
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))

    for dataset in [train_data, val_data, test_data]:
        dataset['highlights'] = dataset['highlights'].apply(clean_text)
        dataset['article'] = dataset['article'].apply(clean_text)

    # Prepare datasets
    train_dataset = TextTokenized(train_data["article"], train_data["highlights"], tokenizer)
    val_dataset = TextTokenized(val_data["article"], val_data["highlights"], tokenizer)
    test_dataset = TextTokenized(test_data["article"], test_data["highlights"], tokenizer)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=1)

    return  train_loader, val_loader, test_loader

def get_ngrams(text, n):
    tokens = text.split()
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def LCS(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if text1[i] == text2[j]:
                dp[i][j] = 1 if (i == 0 or j == 0) else (dp[i-1][j-1] + 1)
            if i > 0 and dp[i-1][j] > dp[i][j]:
                dp[i][j] = dp[i-1][j]
            if j > 0 and dp[i][j-1] > dp[i][j]:
                dp[i][j] = dp[i][j-1]

    return dp[m-1][n-1]


def rouge_merge(generated_summary, reference_summary, n=0):
  if n == 0:
    # Tokenize the summaries
    generated_tokens = generated_summary.split()
    reference_tokens = reference_summary.split()

    # Compute LCS
    count = LCS(generated_tokens, reference_tokens)

  else:
    generated_tokens = get_ngrams(generated_summary, n)
    reference_tokens = get_ngrams(reference_summary, n)

    # Count overlapping n-grams
    overlap = generated_tokens & reference_tokens
    count = len(overlap)

  # Precision and recall
  if len(generated_tokens) == 0:
    precision = 0.0
  else:
    precision = count / len(generated_tokens)


  if len(reference_tokens) == 0:
    recall = 0.0
  else:
    recall = count / len(reference_tokens)

  # F1 score
  if precision + recall == 0:
      return 0.0

  return 2 * (precision * recall) / (precision + recall)


def evaluate_rouge(generated_summaries, reference_summaries, n=0):
    scores = [
        rouge_merge(generated, reference, n)
        for generated, reference in zip(generated_summaries, reference_summaries)
    ]
    return sum(scores) / len(scores) if scores else 0


def testing(model, test_loader, tokenizer, rouge12_request=0, rougel_request=0):
    model.eval()
    generated_summaries = []
    reference_summaries = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()


            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
            )

            batch_predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
            batch_references = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]

            generated_summaries.extend(batch_predictions)
            reference_summaries.extend(batch_references)

    # Compute ROUGE scores
    if rouge12_request == 1:
        rouge_1 = evaluate_rouge(generated_summaries, reference_summaries, n=1)
        rouge_2 = evaluate_rouge(generated_summaries, reference_summaries, n=2)
        print(f"ROUGE-1: {rouge_1:.4f}, ROUGE-2: {rouge_2:.4f}",end=' ')
    if rougel_request == 1:
        rouge_l = evaluate_rouge(generated_summaries, reference_summaries, n=0)
        print(f"ROUGE-l: {rouge_l:.4f}")


def main():

    # test_model = "./pruned_model_pretrained/ratio-0.5_expand"
    # test_model = "./pruned_model_pretrained/ratio-0.25_expand"
    test_model = "../llama-2/7B"
    tokenizer = AutoTokenizer.from_pretrained(test_model)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    model = AutoModelForCausalLM.from_pretrained(
        # 'llama-2/7B',
        test_model,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map='auto',
        # low_cpu_mem_usage=True
    )

    data_path = 'hw2_datasets'
    _, _, test_loader = dataset_load(data_path, tokenizer)
    testing(model, test_loader, tokenizer, rouge12_request=1, rougel_request=1)

if __name__ == '__main__':
    main()