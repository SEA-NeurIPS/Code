import os
import json
import argparse
import numpy as np
import shutil
import subprocess
from evaluation.metric_eval import eval_metric
from evaluation.attack_eval import eval_attack
from evaluation.reason_eval import eval_reason
from evaluation.truth_eval import eval_truth

def options():
    parser = argparse.ArgumentParser()
    ## basic setting
    parser.add_argument("--path", type=str)
    parser.add_argument("--num-sample", type=int)
    parser.add_argument("--key-words", type=str, default=None)
    args = parser.parse_args()
    return args

def get_subdirectories(p, name="output.json", keyword=None):
    if keyword:
        subdirs = [os.path.join(p, d, name) for d in os.listdir(p) 
                   if ((os.path.isdir(os.path.join(p, d))) and (keyword in d) and (os.path.exists(os.path.join(p, d, name))))]
    else:
        subdirs = [os.path.join(p, d, name) for d in os.listdir(p) if ((os.path.isdir(os.path.join(p, d))) and (os.path.exists(os.path.join(p, d, name))))]
    return subdirs

def clean(text):
    if text:
        text=text.strip()
        p1="<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        text = text.split(p1)
        text = list(filter(None, text))[-1]
        text = text.replace("<|eot_id|>", "")
        text = text.replace("Here is the rewritten text:\n\n", "")
        text = text.replace("Here's a rewritten version of the answer:\n\n", "")
        text = text.replace("Here is a rewritten version of the text with improved language and completed sentences:\n\n", "")
        text = text.split("<A>")[-1]
        text = text.split("rewritten answer")[-1]
        text = text.split("Rewritten Answer")[-1]
        text = text.split("revised answer")[-1]
        text = text.split("Revised Answer")[-1]
        text = text.replace("<Q>", "")
        text = text.replace("Here's a rewritten version of the answer under R, ", "")
        text = text.strip()
    return text


def process_json_iea(input_json, path):
    output_data = []

    for item in input_json:

        instruction = eval(item["prompt"])[0]["content"]
        reward_scores = item["optimized_hard_reward_score"]
        revised_reward_scores = item["revised_reward_score"]
        all_scores = np.concatenate((np.array(reward_scores), np.array(revised_reward_scores)))
        all_response = item["optimized_response"] + item["revised_optimized_response"]
        max_index = np.argmax(all_scores)
        output = all_response[max_index]
        output = clean(output)

        reward = np.max(all_scores)
        if reward != all_scores[max_index]:
            raise

        generator = "{}_{}_{}".format(path.split("/")[-5], path.split("/")[-3], path.split("/")[-2])
        
        processed_item = {
            "instruction": instruction,
            "output": output,
            "reward": reward,
            "generator": generator
        }
        output_data.append(processed_item)
    return output_data


def process_json_bon(input_json, path, save_to_ori=False):
    output_data = []

    for item in input_json:

        instruction = eval(item["prompt"])[0]["content"]
        output = item["best_bon_response"]
        output = clean(output)
        reward = item["best_reward_score"]
        if save_to_ori:
            generator = "{}_original_do_sample_true".format(path.split("/")[-5])
        else:
            generator = "{}_{}_{}".format(path.split("/")[-5], path.split("/")[-3], path.split("/")[-2])
        
        processed_item = {
            "instruction": instruction,
            "output": output,
            "reward": reward,
            "generator": generator
        }
        output_data.append(processed_item)
    return output_data


def process_json_args(input_json, path):
    output_data = []
    
    for item in input_json:

        instruction = eval(item["prompt"])[0]["content"]
        output = item["args_response"][0]
        output = clean(output)
        reward = item["reward_score"][0]
        generator = "{}_{}_{}".format(path.split("/")[-5], path.split("/")[-3], path.split("/")[-2])
        
        processed_item = {
            "instruction": instruction,
            "output": output,
            "reward": reward,
            "generator": generator
        }
        output_data.append(processed_item)
    return output_data

def process_json_cbs(input_json, path):
    output_data = []
    
    for item in input_json:

        instruction = eval(item["prompt"])[0]["content"]
        output = item["CBS_response"]
        output = clean(output)
        reward = item["reward_score"]
        generator = "{}_{}_{}".format(path.split("/")[-5], path.split("/")[-3], path.split("/")[-2])
        
        processed_item = {
            "instruction": instruction,
            "output": output,
            "reward": reward,
            "generator": generator
        }
        output_data.append(processed_item)
    return output_data


def process_json_rs(input_json, path):
    output_data = []
    
    for item in input_json:

        instruction = eval(item["prompt"])[0]["content"]
        output = item["rs_response"]
        output = clean(output)
        reward = item["reward_score"]
        generator = "{}_{}_{}".format(path.split("/")[-5], path.split("/")[-3], path.split("/")[-2])
        
        processed_item = {
            "instruction": instruction,
            "output": output,
            "reward": reward,
            "generator": generator
        }
        output_data.append(processed_item)
    return output_data


def process_json_cards(input_json, path):
    output_data = []
    
    for item in input_json:

        instruction = eval(item["prompt"])[0]["content"]
        output = item["CARDS_response"]
        output = clean(output)
        reward = item["reward_score"]
        generator = "{}_{}_{}".format(path.split("/")[-5], path.split("/")[-3], path.split("/")[-2])
        
        processed_item = {
            "instruction": instruction,
            "output": output,
            "reward": reward,
            "generator": generator
        }
        output_data.append(processed_item)
    return output_data


def process_json_original(input_json, path):
    output_data = []

    for item in input_json:

        instruction = eval(item["prompt"])[0]["content"]
        output = item["original_response"]
        output = clean(output)
        reward = item["original_response_reward_score"]
        generator = "{}_original_do_sample_false".format(path.split("/")[-5])
        
        processed_item = {
            "instruction": instruction,
            "output": output,
            "reward": reward,
            "generator": generator
        }
        output_data.append(processed_item)
    return output_data


def process_json_mtbench(input_file):

    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile) 
        
    processed_data = []
    question_id_counter = 81 
    
    for entry in data:
        answer_id = entry.get("answer_id", "Unknown") 
        model_id = entry.get("generator", "Unknown")
        choices = []
        
        choices.append({
            "index": 0,
            "turns": [entry.get("output", []),""] 
        })
        
        processed_data.append({
            "question_id": question_id_counter,
            "answer_id": answer_id,
            "model_id": model_id,
            "choices": choices,
            "tstamp": entry.get("best_iter", [None])[0]  
        })
        
        question_id_counter += 1 

    output_file = os.path.join("/".join(input_file.split('/')[:-1]), "{}.jsonl".format(model_id))

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in processed_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')  

    target_directory = './evaluation/llm_judge/data/mt_bench/model_answer'
    target_file = os.path.join(target_directory, "{}.jsonl".format(model_id))
    shutil.move(output_file, target_file)

    return model_id


def format_output(save_path):
    methods = [ d for d in os.listdir(save_path) if ((os.path.isdir(os.path.join(save_path, d))) and (d in method_list))]
    for method in methods:
        path = os.path.join(save_path, method)
        runs = get_subdirectories(path)
        
        for i, r in enumerate(runs):

            with open(r, 'r', encoding='utf-8') as f:
                input_json = json.load(f)

            if r.split('/')[-3] == 'iea':
                output_file_iea = os.path.join("/".join(r.split('/')[:-1]), 'format_output.json') 
                output_file_ori = os.path.join("/".join(r.split('/')[:-3]), 'ori') 
                os.makedirs(output_file_ori, exist_ok=True)
                output_file_ori = os.path.join(output_file_ori, 'format_output_do_sample_false.json') 

                processed_output = process_json_iea(input_json, r)
                with open(output_file_iea, 'w', encoding='utf-8') as f:
                    json.dump(processed_output, f, indent=2, ensure_ascii=False)

                processed_output = process_json_original(input_json, r)
                with open(output_file_ori, 'w', encoding='utf-8') as f:
                    json.dump(processed_output, f, indent=2, ensure_ascii=False)
            
            elif r.split('/')[-3] == 'bon':
                print('r',r)
                output_file_bon = os.path.join("/".join(r.split('/')[:-1]), 'format_output.json') 
                processed_output = process_json_bon(input_json, r)
                with open(output_file_bon, 'w', encoding='utf-8') as f:
                    json.dump(processed_output, f, indent=2, ensure_ascii=False)

                if '_n1_' in r:
                    output_file_ori = os.path.join("/".join(r.split('/')[:-3]), 'ori') 
                    os.makedirs(output_file_ori, exist_ok=True)
                    output_file_ori = os.path.join(output_file_ori, 'format_output_do_sample_true.json') 
                    processed_output = process_json_bon(input_json, r, save_to_ori=True)
                    with open(output_file_ori, 'w', encoding='utf-8') as f:
                        json.dump(processed_output, f, indent=2, ensure_ascii=False)

            elif r.split('/')[-3] == 'args':
                output_file_args = os.path.join("/".join(r.split('/')[:-1]), 'format_output.json') 
                processed_output = process_json_args(input_json, r)
                with open(output_file_args, 'w', encoding='utf-8') as f:
                    json.dump(processed_output, f, indent=2, ensure_ascii=False)
            
            elif r.split('/')[-3] == 'cbs':
                output_file_args = os.path.join("/".join(r.split('/')[:-1]), 'format_output.json') 
                processed_output = process_json_cbs(input_json, r)
                with open(output_file_args, 'w', encoding='utf-8') as f:
                    json.dump(processed_output, f, indent=2, ensure_ascii=False)

            elif r.split('/')[-3] == 'rs':
                output_file_args = os.path.join("/".join(r.split('/')[:-1]), 'format_output.json') 
                processed_output = process_json_rs(input_json, r)
                with open(output_file_args, 'w', encoding='utf-8') as f:
                    json.dump(processed_output, f, indent=2, ensure_ascii=False)

            elif r.split('/')[-3] == 'cards':
                output_file_args = os.path.join("/".join(r.split('/')[:-1]), 'format_output.json') 
                processed_output = process_json_cards(input_json, r)
                with open(output_file_args, 'w', encoding='utf-8') as f:
                    json.dump(processed_output, f, indent=2, ensure_ascii=False) 
            else:
                raise

def do_eval(save_path, data_limit):
    methods = [ d for d in os.listdir(save_path) if ((os.path.isdir(os.path.join(save_path, d))) and (d in method_list))]
    for method in methods:
        path = os.path.join(save_path, method)
        runs = get_subdirectories(path, name='format_output.json')

        if 'advbench' in path:
            for i, p in enumerate(runs):
                metircs = eval_attack(p, data_limit)
                if metircs is None:
                    metircs = {'reward':'wrong', 'matched_percentage':'wrong',  'longformer_percentage':'wrong',  'longformer_confidence':'wrong'}

                if i == 0:
                    print("\nBase Model\t\tMethod\t\tRun\t\tReward\t\tMatch(%)\t\tLF-ASR(%)\t\tLF-Conf(%)")
                if metircs:
                    print('-'*200)
                    _p="\t".join([p.split('/')[-5],p.split('/')[-3],p.split('/')[-2]])
                    print("{}\n{}\t{}\t{}\t{}".format(_p, metircs["reward"], metircs["matched_percentage"], metircs["longformer_percentage"], metircs["longformer_confidence"]))
        elif 'gsm8k' in path or 'math' in path:
            for i, p in enumerate(runs):
                metircs = eval_reason(p, data_limit)
                if metircs is None:
                    metircs = {'accuracy':'wrong', 'reward':'wrong'}

                if i == 0:
                    print("\nBase Model\t\tMethod\t\tRun\t\tReward\t\tAccuracy(%)")
                if metircs:
                    print('-'*200)
                    _p="\t".join([p.split('/')[-5],p.split('/')[-3],p.split('/')[-2]])
                    print("{}\n{}\t{}".format(_p, metircs["reward"], metircs["accuracy"]))
                    print('\n')
        elif 'alpacaeval2' in path:
            for i, p in enumerate(runs):
                metircs = eval_metric(p, data_limit)
                if metircs is None:
                    metircs = {'diversity':'wrong', 'coherence':'wrong', 'reward':'wrong'}

                ref_path = './data/alpaca/alpaca_farm_evaluation.json'
                output_path= os.path.join('/'.join(p.split('/')[:-1]), 'alpaca_eval', 'vs_chosen')
                command = [p, ref_path, output_path, str(data_limit)]
                results = subprocess.run(['bash', "./evaluation/alpaca_eval.sh"] + command, capture_output=True, text=True)
                split_list = results.stdout.split(p.split('/')[-2])[-1].split()
                alpaca_vsChosen = [float(num) for num in split_list]
                if len(alpaca_vsChosen)!=5:
                    alpaca_vsChosen = ['wrong', 'wrong', 'wrong', 'wrong', 'wrong']

                ref_path = "./data/alpaca/reference_outputs.json"
                output_path= os.path.join('/'.join(p.split('/')[:-1]), 'alpaca_eval', 'vs_gpt4')
                command = [p, ref_path, output_path, str(data_limit)]
                results = subprocess.run(['bash', "./evaluation/alpaca_eval.sh"] + command, capture_output=True, text=True)
                split_list = results.stdout.split(p.split('/')[-2])[-1].split()
                alpaca_vsGPT4 = [float(num) for num in split_list]
                if len(alpaca_vsGPT4)!=5:
                    alpaca_vsGPT4 = ['wrong', 'wrong', 'wrong', 'wrong', 'wrong']

                ref_path = os.path.join("/".join(p.split('/')[:-3]), 'ori', 'format_output_do_sample_true.json') 
                output_path= os.path.join('/'.join(p.split('/')[:-1]), 'alpaca_eval', 'vs_ori')
                command = [p, ref_path, output_path, str(data_limit)]
                results = subprocess.run(['bash', "./evaluation/alpaca_eval.sh"] + command, capture_output=True, text=True)
                split_list = results.stdout.split(p.split('/')[-2])[-1].split()
                alpaca_vsORI = [float(num) for num in split_list]
                if len(alpaca_vsORI)!=5:
                    alpaca_vsORI = ['wrong', 'wrong', 'wrong', 'wrong', 'wrong']                

                if i == 0:
                    print("\nBase Model\t\tMethod\t\tRun\t\tDiversity\t\tCoherence\t\tReward\t\tLC\t\tWR\t\tStd\t\tN Total\t\tAvg Length")
                if metircs:
                    print('-'*200)
                    _p="\t".join([p.split('/')[-5],p.split('/')[-3],p.split('/')[-2]])
                    print("{}\n{}\t{}\t{}\nvs GPT4\t{}\t{}\t{}\t{}\t{}\nvs ORI\t{}\t{}\t{}\t{}\t{}\nvs Chosen\t{}\t{}\t{}\t{}\t{}".format(_p, metircs["diversity"], metircs["coherence"], metircs["reward"], 
                    *alpaca_vsGPT4,
                    *alpaca_vsORI,
                    *alpaca_vsChosen
                    ))
        elif 'mtbench' in path:
            
            for i, p in enumerate(runs):
                model_id = process_json_mtbench(p)
                print("\nBase Model\t\tMethod\t\tRun\t\tDiversity\t\tCoherence\t\tReward")
                print('-'*200)
                command = [
                    "python", "gen_judgment.py",  # Python executable and script
                    "--model-list", model_id,     # Pass the model_id to --model-list
                    "--parallel", "1",           # Set parallel to 10
                    "--first-n", str(data_limit)             # Set first-n to 10
                ]
                results=subprocess.run(command, cwd='evaluation/llm_judge')

                command = [
                    "python", "show_result.py",  # Python executable and script
                    "--model-list", model_id,     # Pass the model_id to --model-list
                ]
                results=subprocess.run(command, cwd='evaluation/llm_judge')

                metircs = eval_metric(p, data_limit)
                if metircs is None:
                    metircs = {'diversity':'wrong', 'coherence':'wrong', 'reward':'wrong'}
                _p="\t".join([p.split('/')[-5],p.split('/')[-3],p.split('/')[-2]])
                print("{}\n{}\t{}\t{}".format(_p, metircs["diversity"], metircs["coherence"], metircs["reward"]))

        elif 'tqa' in path:
            
            for i, p in enumerate(runs):
                print("\nBase Model\t\tMethod\t\tRun\t\tReward\t\tTruth\t\tInfo\t\tDiversity\t\tCoherence")
                print('-'*200)
                truth_metircs = eval_truth(p, data_limit)
                if truth_metircs is None:
                    truth_metircs = {'truth':'wrong', 'info':'wrong', 'reward':'wrong'}
                metircs = eval_metric(p, data_limit)
                if metircs is None:
                    metircs = {'diversity':'wrong', 'coherence':'wrong', 'reward':'wrong'}
                _p="\t".join([p.split('/')[-5],p.split('/')[-3],p.split('/')[-2]])
                print("{}\n{}\t{}\t{}\t{}\t{}".format(_p, truth_metircs["reward"], truth_metircs["truth"], truth_metircs["info"], metircs["diversity"], metircs["coherence"]))
        else:
            raise

def main():
    args = options()
    format_output(args.path)
    print("format done\n")
    do_eval(args.path, args.num_sample)

method_list=["bon","iea","args","cbs","cards"]
main()