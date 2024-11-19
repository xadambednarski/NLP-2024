import torch
from transformers import AutoTokenizer, BertTokenizer, BertForMaskedLM
import random
import numpy as np
from datasets import load_dataset
import json
import re

data_files = {"train": "train.jsonl", "test": "test.jsonl"}
dataset = load_dataset("./sentences", data_files=data_files)

tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = BertForMaskedLM.from_pretrained("allegro/herbert-base-cased")
model.eval()

sentences = dataset["train"]["text"]


def clean_token_artifacts(text):
    text = re.sub(r'\[CLS\]|\[SEP\]', '', text).strip()
    text = re.sub(r' ##', '', text)
    text = text.replace('[UNK]', '')
    text = re.sub(r'\s([?.!",](?:\s|$))', r'\1', text)
    text = text.replace("@ anonymized _ account", "@anonymized_account")
    return text


def mask_words(sentences, tokenizer, mlm_probability=0.15):
    masked_sentences = []
    for sentence in sentences:
        words = sentence.split()
        num_to_mask = int(len(words) * mlm_probability)
        mask_indices = np.random.choice(len(words), num_to_mask, replace=False)
        for index in mask_indices:
            words[index] = tokenizer.mask_token
        masked_sentence = ' '.join(words)
        masked_sentences.append(masked_sentence)
    return masked_sentences


masked_sentences = mask_words(sentences, tokenizer)

inputs = tokenizer(masked_sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

predicted_sentences = []
for input_ids, pred_ids in zip(inputs.input_ids, predictions):
    pred_tokens = [tokenizer.decode([pid]) if iid == tokenizer.mask_token_id else tokenizer.decode([iid]) for iid, pid in zip(input_ids, pred_ids)]
    predicted_sentence = ' '.join(pred_tokens).replace(tokenizer.pad_token, '')
    predicted_sentences.append(predicted_sentence)


with open("./sentences/train_with_masks.jsonl", 'w') as f:
    for sentence, label in zip(predicted_sentences, dataset["train"]["label"]):
        item = {"text": clean_token_artifacts(sentence.strip()), "label": label}
        f.write(json.dumps(item) + '\n')

    for item in dataset["train"]:
        f.write(json.dumps(item) + '\n')

print("Predictions saved to train_with_masks.jsonl")
