import os
import re
import sys
import json
import time
import math
import torch
import random
import transformers
import numpy as np
import argparse
from datasets import load_dataset
import traceback
import operator
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from .utils import evaluate_math

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def eval_math(path, data_limit=None):
    outputs = []
    answers = []
    rewards = []
    types = []
    levels = []

    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = 0
    total = 0

    completions = json.load(open(path, "r"))
    if data_limit and (len(completions) < data_limit):
        print('-'*150)
        print(f"{path}: Not Enough Sample")
        return None
    completions = completions[:data_limit]
    all_problems = load_dataset("HuggingFaceH4/MATH-500", split='test')

    for problem_data, model_output in tqdm(zip(all_problems, completions), total=len(all_problems), desc="Matching"):
        prob_level = int(problem_data["level"])
        prob_type = problem_data["subject"]
        answer = problem_data["answer"]
        reward = model_output['reward']

        is_matched, equiv, model_output = evaluate_math(model_output["output"], answer)
        levels.append(prob_level)
        types.append(prob_type)
        outputs.append(model_output)
        answers.append(answer)
        rewards.append(reward)

        fnames_list.append(equiv)
        if (prob_level, prob_type) in cors:
            cors[(prob_level, prob_type)].append(equiv)
        else:
            cors[(prob_level, prob_type)] = [equiv]
        if prob_level in level_cors:
            level_cors[prob_level].append(equiv)
        else:
            if prob_level is not None:
                level_cors[prob_level] = [equiv]
        if prob_type in subject_cors:
            subject_cors[prob_type].append(equiv)
        else:
            if prob_type is not None:
                subject_cors[prob_type] = [equiv]
        if equiv:
            correct += 1
    
    save_dir='/'.join(path.split('/')[:-1])
    output_file = os.path.join(save_dir, "results.txt")
    
    output_dict = {
        "outputs": [],
        "accuracy_by_subject_and_level": defaultdict(list),
        "accuracy_by_level": [],
        "accuracy_by_subject": [],
    }
    with open(output_file, "w+") as f:
        for k, (output, answer, prob_type, prob_level, equiv) in enumerate(zip(outputs, answers, types, levels, fnames_list)):
            f.write("{} TYPE: {} | LEVEL: {} | OUTPUT: {} | ANSWER: {} | CORRECT: {}\n".format(k, prob_type, prob_level, output, answer, equiv))
            output_dict["outputs"].append({
                "type": prob_type,
                "level": prob_level,
                "output": output,
                "answer": answer,
                "equiv": equiv
            })

        f.write("#####################\n")
        # also get accuracies for each
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            for level in range(1, 6):
                key = (level, subject)
                if key not in cors.keys():
                    # print("Skipping", key)
                    continue
                cors_list = cors[key]
                # print("{} Level {} Accuracy = {}/{} = {:.3f}".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                f.write("{} Level {} Accuracy = {}/{} = {:.3f}\n".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                
                output_dict["accuracy_by_subject_and_level"][subject].append({
                    "level": level,
                    "num_correct": np.sum(cors_list),
                    "num_total": len(cors_list),
                    "accuracy": np.mean(cors_list)
                })

        # print("#####################")
        f.write("#####################\n")
        for level in sorted(level_cors):
            if level not in level_cors.keys():
                print("Skipping", level)
                continue
            cors_list = level_cors[level]
            # print("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("Level {} Accuracy = {}/{} = {:.3f}\n".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            output_dict["accuracy_by_level"].append({
                "level": level,
                "num_correct": np.sum(cors_list),
                "num_total": len(cors_list),
                "accuracy": np.mean(cors_list)
            })

        # print("#####################")
        f.write("#####################\n")
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            if subject not in subject_cors.keys():
                # print("Skipping", subject)
                continue
            cors_list = subject_cors[subject]
            print("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("{} Accuracy = {}/{} = {:.3f}\n".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            output_dict["accuracy_by_subject"].append({
                "subject": subject,
                "num_correct": np.sum(cors_list),
                "num_total": len(cors_list),
                "accuracy": np.mean(cors_list)
            })
        # print("#####################")
        f.write("#####################\n")
        total = len(answers)
        # print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct/total * 100))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct/total * 100))
        output_dict["overall_accuracy"] = {
            "num_correct": correct,
            "num_total": total,
            "reward": float(sum(rewards)) / len(rewards),
            "accuracy": (correct/total) * 100
        }
        class JSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.int64):
                    return int(obj)
                return super(JSONEncoder, self).default(obj)
        with open(os.path.join(save_dir, "results.json"), "w") as jf:
            json.dump(output_dict, jf, cls=JSONEncoder)
    
    return output_dict["overall_accuracy"]


def eval_gsm(path, data_limit=None):

    generations = json.load(open(path, "r"))
    if data_limit and (len(generations) < data_limit):
        print('-'*150)
        print(f"{path}: Not Enough Sample")
        return None
    generations = generations[:data_limit]

    data = load_dataset("openai/gsm8k", 'main', split='train')

    answers = []
    rewards = []
    for sample, completion in tqdm(zip(data, generations)):
        reward = completion['reward']
        model_answer = clean_answer(completion['output'])
        is_cor = is_correct(model_answer, sample["answer"])
        answers.append(is_cor)
        rewards.append(reward)

    evaluation = {
        "num_sample":len(answers),
        "num_correct":sum(answers),
        "accuracy":(float(sum(answers)) / len(answers)) * 100,
        "reward": float(sum(rewards)) / len(rewards),
    }
    return evaluation


def eval_reason(path, data_limit=None):
    if path.split('/')[-6] == 'gsm8k':
        return eval_gsm(path=path, data_limit=data_limit)
    elif path.split('/')[-6] == 'math':
        return eval_math(path=path, data_limit=data_limit)
    else:
        raise
