import random
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from datasets import load_dataset

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.config.pad_token_id = model.config.eos_token_id  # Avoids some warnings

data_files = {"train": "train.jsonl", "test": "test.jsonl"}
dataset = load_dataset("./sentences", data_files=data_files)

sentences = dataset["train"]["text"]

predicted_sentences = []

def remove_random_word(sentence):
    words = sentence.split()
    if len(words) < 2:
        return None, None 
    remove_index = random.randint(0, len(words) - 1)
    removed_word = words.pop(remove_index)
    truncated_sentence = " ".join(words)
    return truncated_sentence, removed_word, remove_index

# Loop through each sentence
for sentence in sentences:
    truncated_sentence, original_word, remove_index = remove_random_word(sentence)
    if truncated_sentence is None:
        continue  # Skip if sentence couldn't be processed

    # Encode the truncated sentence
    input_ids = tokenizer.encode(truncated_sentence, return_tensors="pt")

    # Generate predictions for the next word
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 1,  # Predict the next word only
            num_return_sequences=1,
            pad_token_id=model.config.eos_token_id,
            do_sample=False,  # Greedy decoding for the most likely word
        )

    predicted_word_id = outputs[0, input_ids.shape[1]:]
    predicted_word = tokenizer.decode(predicted_word_id, skip_special_tokens=True).strip()

    reconstructed_sentence = truncated_sentence.split()
    reconstructed_sentence.insert(remove_index, predicted_word)
    reconstructed_sentence = " ".join(reconstructed_sentence)

    predicted_sentences.append({
        "original_sentence": sentence,
        "truncated_sentence": truncated_sentence,
        "original_word": original_word,
        "predicted_word": predicted_word,
        "reconstructed_sentence": reconstructed_sentence
    })

# Save to a CSV file
df = pd.DataFrame(predicted_sentences)
df.to_csv("predicted_sentences_gpt2.csv", index=False)

print("Predicted sentences saved to 'predicted_sentences_gpt2.csv'")
