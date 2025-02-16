from tqdm import tqdm
import json
import torch
import argparse
from nltk import word_tokenize
import os
import numpy as np
import nltk
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')

def get_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(bottleneck=True)
    parser.set_defaults(augment=True)
    args = parser.parse_args()

    return args


def compute_rep_n(text, n):
    tokens = word_tokenize(text)
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    rep_n = 100 * (1.0 - len(set(ngrams)) / (len(ngrams) + 1))
    return rep_n


def compute_diversity(text):
    diversity = 1.0
    for n in range(2, 5):
        rep_n_val = compute_rep_n(text, n)
        diversity *= 1.0 - rep_n_val / 100
    return diversity


def clean(text):
    return text.replace('<|eot_id|>', "")


def average(entries):
    return sum(entries) / len(entries)


def similarity(model, tokenizer, queries, keys, device: str = None):

    queries = tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        query_vecs = model(**queries, output_hidden_states=True, return_dict=True).pooler_output#.to(model.device)

    keys = tokenizer(keys, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        key_vecs = model(**keys, output_hidden_states=True, return_dict=True).pooler_output#.to(model.device)

    # query_vecs = self.encode(queries, device=device, return_numpy=True) # suppose N queries
    
    # if not isinstance(keys, ndarray):
    #     key_vecs = self.encode(keys, device=device, return_numpy=True) # suppose M keys
    # else:
    #     key_vecs = keys

    # check whether N == 1 or M == 1
    single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1 
    if single_query:
        query_vecs = query_vecs.reshape(1, -1)
    if single_key:
        key_vecs = key_vecs.reshape(1, -1)
    
    # returns an N*M similarity array
    similarities = cosine_similarity(query_vecs.cpu().numpy(), key_vecs.cpu().numpy())
    
    if single_query:
        similarities = similarities[0]
        if single_key:
            similarities = float(similarities[0])
    
    return similarities

def compute_coherence(prompts, responses):
    # model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    # similarities = np.array(model.similarity(prompts, responses))
    # return similarities.trace() / len(similarities)
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", device_map='auto')#.cuda()
    similarities = similarity(model, tokenizer, prompts, responses)
    return similarities.trace() / len(similarities)


def eval_metric(path, data_limit=None):
    generations = json.load(open(path, "r"))
    if data_limit and (len(generations) < data_limit):
        print('-'*150)
        print(f"{path}: Not Enough Sample")
        return None
    generations = generations[:data_limit]

    entries = []
    for generation in generations:
        
        prompt = generation["instruction"]
        response = generation["output"]
        reward = generation["reward"]

        
        if len(response) == 0:
            response = " "
        rep_2 = compute_rep_n(response, 2)
        rep_3 = compute_rep_n(response, 3)
        rep_4 = compute_rep_n(response, 4)
        diversity = compute_diversity(response)
        # print(generation.keys())
        entries.append(
            {
                "prompt": prompt,
                "response": response,
                "reward": reward,
                # "original_response": generation["response"][len(prompt) :],
                "rep_2": rep_2,
                "rep_3": rep_3,
                "rep_4": rep_4,
                "diversity": diversity,
                "response_length": len(response),
                # "elapsed": generation["elapsed"],
            }
        )

    evaluations = {
        "rep_2": average([entry["rep_2"] for entry in entries]),
        "rep_3": average([entry["rep_3"] for entry in entries]),
        "rep_4": average([entry["rep_4"] for entry in entries]),
        "diversity": average([entry["diversity"] for entry in entries]),
        "coherence": compute_coherence(
            [entry["prompt"] for entry in entries], [entry["response"] for entry in entries]
        ),
        "reward": average([entry["reward"] for entry in entries]),
        "response_length": average([entry["response_length"] for entry in entries]),
        # "elapsed": average([entry["elapsed"] for entry in entries]),
        "entries": entries,
    }

    # create the evaluations directory if it does not exist
    eval_path = os.path.join("/".join(path.split('/')[:-1]), "metric_eval.json") #os.path.join("evaluations", f"{args.run_name}.json")
    json.dump(evaluations, open(eval_path, "w"), indent=2)
    return evaluations


if __name__ == "__main__":
    path = ''
    main(path,2)
