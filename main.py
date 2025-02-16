#!/usr/bin/env python
# coding: utf-8

import os
import time
import wandb
import logging
import argparse
import random
import sys
import numpy as np
import pandas as pd
from datetime import timedelta

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import gather_object, InitProcessGroupKwargs

from util import *
from model import *
from decoding import decode
from baseline import bon, args_1, cbs

logger = logging.getLogger(__name__)

def options():
    parser = argparse.ArgumentParser()
    ## basic setting
    parser.add_argument("--mode", type=str, default='iea')
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--debug", action="store_true")

    ## analysis mode
    parser.add_argument("--jail-break", type=int, choices=[1,3,5,7,9], default=None)
    parser.add_argument("--token-kl", action="store_true")

    ## save paths
    parser.add_argument("--save-path", default="outputs")
    parser.add_argument("--save-file-name", default="output.json")
    
    
    ## device & dtype
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--flash-atten-2", action="store_true")
    parser.add_argument("--no-cuda", action="store_true", help="no cuda")
    parser.add_argument("--accelerator", action="store_true")
    parser.add_argument("--ddp", action="store_true")

    ## data setting
    parser.add_argument("--dataset", type=str,default="hh")
    parser.add_argument("--start", type=int, default=-1, help="loading data from ith examples.")
    parser.add_argument("--end", type=int, default=1000000000000, help="loading data util ith examples.")
    ## model setting
    parser.add_argument("--reward-model", type=str, default=None)
    parser.add_argument("--pretrained-model", type=str, default="mistral-7b")
    ## training setting
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--outter-batch-size", type=int, default=2)
    parser.add_argument("--straight-through", action="store_true")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--reward-topk", type=int, default=0)
    parser.add_argument("--div", type=str, default='FKL')
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--length", type=int, default=128, help="maximum length of optimized logits.")
    parser.add_argument("--min-length", type=int, default=50, help="min length of complete sentence.")
    parser.add_argument("--max-length", type=int, default=2048, help="maximum length of complete sentence.")
    parser.add_argument("--frozen-length", type=int, default=0, help="length of optimization window in sequence.")
    parser.add_argument("--revise-mode", type=str, default='QA', choices=['A', 'QA-ICL', 'QA'])
    
    # weight
    parser.add_argument("--reward-weight", type=float, default=1)
    parser.add_argument("--flu-weight", type=float, default=1)
    parser.add_argument("--end-weight", type=float, default=1)
    # temperature
    parser.add_argument("--reward-temp", type=float, default=0.2, help="temperature of logits used in reward model.")
    parser.add_argument("--input-lgt-temp", type=float, default=1, help="temperature of logits used for model input.")
    parser.add_argument("--output-lgt-temp", type=float, default=1, help="temperature of logits used for model output.")
    parser.add_argument("--init-temp", type=float, default=0.1, help="temperature of logits used in the initialization pass. High => uniform init.")
    parser.add_argument("--init-mode", type=str, default='original', choices=['random', 'original', 'prompt', 'bon'])
    # lr
    parser.add_argument("--optim", type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate in the backward pass.")
    parser.add_argument("--scheduler-stepsize", type=int, default=200, help="")
    parser.add_argument("--scheduler-gamma", type=float, default=1, help="")
    # iterations
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--min-iters", type=int, default=0, help="record best only after N iterations")
    parser.add_argument("--noise-iters", type=int, default=1, help="add noise at every N iterations")
    parser.add_argument("--win-anneal-iters", type=int, default=-1, help="froze the optimization window after N iters")
    # gaussian noise
    parser.add_argument("--gs-mean", type=float, default=0.0)
    parser.add_argument("--gs-std", type=float, default=0.01)
    parser.add_argument("--large-noise-iters", type=str, default="-1", help="Example: '50,1000'")
    parser.add_argument("--large-gs-std", type=str, default="1", help="Example: '1,0.1'")

    # baseline setting
    parser.add_argument("--n-of-bon", type=int, default=128)
    parser.add_argument("--top-k", type=float, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--args-weight", type=float, default=1.0)
    parser.add_argument("--args-mode", type=str, default='greedy', choices=['greedy', 'categorical'])
    parser.add_argument("--not-do-sample", action="store_true")
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--successors-per-state", type=int, default=4)
    parser.add_argument("--chunk-length", type=int, default=32)


    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def collate_fn(examples):
    prompts, chosens, rejecteds, datasets = [], [], [], []
    if ('goal' in examples[0]) and ('target' in examples[0]): #AdvBench
        for ex in examples:
            rejected = "Human:{}\n\nAssistant:{}".format(ex["goal"],ex["target"])
            if not is_openai_format(rejected):
                rejected_parsed = parse_conversation(rejected)
            else:
                rejected_parsed = rejected

            if len(rejected_parsed) > 1:
                prompts.append(rejected_parsed[:-1])
                rejecteds.append(rejected_parsed)
                chosens.append(None)
                datasets.append(None)
            else:
                continue    
    elif 'instruction' in examples[0]: # AlpacaEval2
        for ex in examples:
            prompt = [{"role":"user", "content":ex['instruction']}]
            prompts.append(prompt)
            datasets.append(ex["dataset"])
            chosens.append(None)
            rejecteds.append(None)
    elif 'turns' in examples[0]: # MTBench
        for ex in examples:
            prompt = [{"role":"user", "content":ex['turns'][0]}]
            prompts.append(prompt)
            datasets.append(ex["category"])
            chosens.append(None)
            rejecteds.append(None)
    elif 'prompt_type' in examples[0]: #Xstext
        for ex in examples:
            if ex["prompt_type"] == 'prompt_safe':
                prompt = [{"role":"user", "content":ex['prompt']}]
                prompts.append(prompt)
                datasets.append(ex["prompt_type"])
                chosens.append(None)
                rejecteds.append(None)
            else:
                pass
    elif 'best_answer' in examples[0]: # TQA
        for ex in examples:
            chosen = "Human:{}\n\nAssistant:{}".format(ex["question"],ex["best_answer"])
            if not is_openai_format(chosen):
                chosen_parsed = parse_conversation(chosen)
            else:
                chosen_parsed = chosen

            rejected = "Human:{}\n\nAssistant:{}".format(ex["question"],ex["incorrect_answers"][0])
            if not is_openai_format(rejected):
                rejected_parsed = parse_conversation(rejected)
            else:
                rejected_parsed = rejected
                
            prompt = [{"role":"user", "content":ex['question']}]
            prompts.append(prompt)
            datasets.append(None)
            chosens.append(chosen_parsed)
            rejecteds.append(rejected_parsed)
    else:
        for ex in examples:
            if ('question' in ex) and ('answer' in ex): # GSM8K
                ex["chosen"] = "Human:{}\n\nAssistant:{}".format(ex["question"],ex["answer"])
            chosen = ex["chosen"]
            # print('chosen',chosen)

            if not is_openai_format(chosen):
                chosen_parsed = parse_conversation(chosen)
            else:
                chosen_parsed = chosen

            if len(chosen_parsed) > 1:
                prompts.append(chosen_parsed[:-1])
                chosens.append(chosen_parsed)
                rejecteds.append(None)
                datasets.append(None)

            else:
                continue
    # print(prompts)
    # print(chosens)
    return {
        "prompts": prompts,
        "chosens": chosens,
        "rejecteds": rejecteds,
        "datasets":datasets
    }

def main():
    args = options()

    if (args.outter_batch_size ==1 and args.ddp) or (args.ddp and (args.outter_batch_size != torch.cuda.device_count())) or (not args.ddp and args.outter_batch_size != 1):
        raise ValueError("Invalid configuration: With DDP, outter_batch_size cannot be 1, and it must match the number of GPUs. Without DDP, outter_batch_size must be 1.")

    if ((args.jail_break is not None) and (args.ddp)) or (args.token_kl and args.ddp):
        raise ValueError("Analysis on jail-breaking & per-token-kl only support Non-DDP mode")

    if args.seed != -1:
        seed_everything(args.seed)

    if args.accelerator:
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        accelerator = Accelerator(kwargs_handlers=[kwargs])
        device = {"": accelerator.process_index}
    else:
        accelerator = None
        device = "auto" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    ### Load pretrained model
    model_path = MODEL_PATH_DICT[args.pretrained_model]
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        device=device, 
        args=args
    )

    reward_model, reward_tokenizer = None, None
    if args.reward_model is not None:
        reward_model_path = MODEL_PATH_DICT[args.reward_model]
        reward_model, reward_tokenizer = load_reward_model_and_tokenizer(
            model_path=reward_model_path,
            device=device, 
            args=args
        )

    if args.accelerator:
        device = accelerator.device
    else:
        device = model.device

    if args.mode =='iea':
        run_name = "{}_bs{}-{}_{}_len{}-max{}-min{}_e{}_{}-lr{}-{}-{}_weight-f{}-r{}-e{}_topk{}-r{}_temp-i{}-r{}_re{}".format(
            args.reward_model, args.batch_size, args.init_mode, args.div, 
            args.length, args.max_length, args.min_length, 
            args.num_iters, args.optim, args.lr, args.scheduler_stepsize, args.scheduler_gamma,
            args.flu_weight, args.reward_weight, args.end_weight, 
            args.topk, args.reward_topk, 
            args.init_temp, args.reward_temp,
            args.revise_mode
        )
    elif args.mode =='bon':
        run_name = "{}_n{}_k{}_p{}_t{}_len{}".format(
                args.reward_model, args.n_of_bon, args.top_k, args.top_p, args.temperature, args.max_length
        )
    elif args.mode =='args':
        run_name = "{}_w{}_k{}_p{}_t{}_mode-{}_len{}".format(
            args.reward_model, args.args_weight, args.top_k, args.top_p, args.temperature, args.args_mode, args.max_length 
        )
    elif args.mode =='cbs':
        run_name = "{}_w{}_k{}_c{}_t{}_len{}".format(
            args.reward_model, args.beam_width, args.successors_per_state, args.chunk_length, args.temperature, args.max_length
        )
    elif args.mode =='ori':
        run_name = "len{}".format(
                args.max_length
        )

    if args.description and args.description != 'none':
        run_name = "{}_{}".format(args.description, run_name)

    fw = os.path.join(args.save_path, args.dataset, args.pretrained_model, args.reward_model, args.mode, run_name)
    
    if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
        if not os.path.exists(fw):
            os.makedirs(fw)
        logging.basicConfig(level=logging.INFO, filename=f'{fw}/{run_name}.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.info(f"Arguments: {vars(args)}")

    ### freeze weights
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if args.dataset == 'hh-helpful':
        data = load_dataset(DATA_PATH_DICT[args.dataset], data_dir="helpful-base", split='test')
    elif args.dataset == 'hh-harmless':
        data = load_dataset(DATA_PATH_DICT[args.dataset], data_dir="harmless-base", split='test')
    elif args.dataset == 'ultrafeedback':
        data = load_dataset(DATA_PATH_DICT[args.dataset], split='test_prefs')
    elif args.dataset == 'advbench':
        data = load_dataset('csv', data_files='data/advbench/harmful_behaviors.csv',  split='train')
    elif args.dataset == 'gsm8k':
        data = load_dataset(DATA_PATH_DICT[args.dataset], 'main', split='train')
    elif args.dataset == 'alpacaeval2':
        #data = load_dataset('tatsu-lab/alpaca_farm', split='eval')
        data = load_dataset('json', data_files='data/alpaca/alpaca_farm_evaluation.json', split='train')
    elif args.dataset == 'mtbench':
        data = load_dataset('json', data_files='data/mtbench.jsonl', split='train')
    elif args.dataset == 'xstest':
        data = load_dataset('csv', data_files='data/xstest.csv', split='train')
    elif args.dataset == 'tqa':
        data = load_dataset(DATA_PATH_DICT[args.dataset], "generation", split='validation')
    elif args.dataset == 'math':
        data = load_dataset(DATA_PATH_DICT[args.dataset], split='test')
        data = data.rename_column('answer','final_answer')
        data = data.rename_column('problem','question')
        data = data.rename_column('solution','answer')
    else:
        pass
    data = data.select(range(max(0,args.start), min(len(data), args.end)))

    if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
        print('data', data)

    dataloader = DataLoader(data, batch_size=args.outter_batch_size, shuffle=False, collate_fn=collate_fn)

    output_json_path = os.path.join(fw, args.save_file_name) #'output.json'
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = []

    # Convert existing data to a set of prompts for easy checking
    existing_prompts = {entry['prompt'] for entry in all_data}

    existing_prompts=[]
    for entry in all_data:
        if eval(entry['prompt'])[0]['role']=='system':
            existing_prompts.append(str(eval(entry['prompt'])[1:]))
        else:
            existing_prompts.append(entry['prompt'])

    if args.accelerator and args.ddp:
        for batch_idx, batch_all in enumerate(dataloader):
            accelerator.wait_for_everyone()
            with accelerator.split_between_processes(batch_all) as batch:
                data_list=[]
                prompts = batch["prompts"]
                chosens = batch["chosens"]
                rejecteds = batch["rejecteds"]
                datasets = batch["datasets"]
                
                if len(prompts) == 0:
                    continue

                # Skip entire batch if all prompts are already processed
                batch_prompts, batch_chosens, batch_rejecteds = [],[],[]
                for idx, prompt in enumerate(prompts):
                    if str(prompt) not in existing_prompts:
                        batch_prompts.append(prompt)
                        batch_chosens.append(chosens[idx])
                        batch_rejecteds.append(rejecteds[idx])

                if  len(batch_prompts) == 0:
                    print(f"GPU {accelerator.process_index}, Continue")
                    batch_results = []
                elif len(batch_prompts) < len(prompts):
                    print("GPU {}, Skip {}".format(accelerator.process_index, len(prompts)-len(batch_prompts)))

                if args.mode == 'iea':
                    # Run batch decoding
                    if len(batch_prompts) > 0:
                        batch_results, elapsed_time = decode(
                            model=model, tokenizer=tokenizer, reward_model=reward_model,reward_tokenizer=reward_tokenizer, 
                            device=device, prompt=batch_prompts, chosen=batch_chosens, rejected=batch_rejecteds,
                            args=args, save_path=fw, run_name=run_name, accelerator=accelerator
                        )
                    for i, (prompt, result) in enumerate(zip(batch_prompts, batch_results)):
                        # print('result',result)
                        data = {
                            'prompt': str(prompt),
                            'chosen_given': result['chosen_given'],
                            'chosen_given_reward_score': result['chosen_given_reward_score'],
                            'rejected_given': result['rejected_given'],
                            'rejected_given_reward_score': result['rejected_given_reward_score'],
                            'original_response': result['original_output'],
                            'original_response_reward_score': result['original_output_reward_score'],
                            'optimized_response': result['text'],
                            'optimized_soft_reward_score': result['soft_reward_score'],
                            'optimized_hard_reward_score': result['hard_reward_score'],
                            'revised_optimized_response': result['revised_text'],
                            'revised_reward_score': result['revised_reward_score'],
                            'best_iter': result['best_iter'],
                            'time':elapsed_time
                        }
                        data_list.append(data)
                
                elif args.mode == 'args':
                    if len(batch_prompts) > 0:
                        batch_results, elapsed_time = args_1(
                            model=model, tokenizer=tokenizer, reward_model=reward_model,reward_tokenizer=reward_tokenizer, 
                            device=device, prompt=batch_prompts, chosen=batch_chosens, rejected=batch_rejecteds,
                            args=args, save_path=fw, run_name=run_name, accelerator=accelerator
                        )
                    for i, (prompt, result) in enumerate(zip(batch_prompts, batch_results)):
                        # print('result',result)
                        data = {
                            'prompt': str(prompt),
                            'args_response': result['response'],
                            'reward_score': result['reward_score'],
                            'time':elapsed_time
                        }
                        data_list.append(data)
                else:
                    raise

            results_gathered = gather_object(data_list)
            results_gathered = merge_duplicates(args.mode, results_gathered)

            if args.accelerator and accelerator.is_main_process:
                all_data.extend(results_gathered)
                with open(os.path.join(fw, args.save_file_name), 'w') as f:
                    json.dump(all_data, f, indent=4)

    else:
        for batch_idx, batch in enumerate(dataloader):
            prompts = batch["prompts"]
            chosens = batch["chosens"]
            rejecteds = batch["rejecteds"]
            datasets = batch["datasets"]
            
            if len(prompts) == 0:
                continue

            # Skip entire batch if all prompts are already processed
            batch_prompts, batch_chosens, batch_rejecteds = [],[],[]
            for idx, prompt in enumerate(prompts):
                if str(prompt) not in existing_prompts:
                    batch_prompts.append(prompt)
                    if len(chosens) > 0:
                        batch_chosens.append(chosens[idx])
                    if len(rejecteds) > 0:
                        batch_rejecteds.append(rejecteds[idx])
            
            if  len(batch_prompts) == 0:
                if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
                    print("Continue")
                continue  # Skip if all prompts are already processed
            elif len(batch_prompts) < len(prompts):
                if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
                    print("Skip {}".format(len(prompts)-len(batch_prompts)))

            if args.mode == 'iea':
                # Run batch decoding
                res_iea, elapsed_time = decode(
                    model=model, tokenizer=tokenizer, reward_model=reward_model,reward_tokenizer=reward_tokenizer, 
                    device=device, prompt=batch_prompts, chosen=batch_chosens, rejected=batch_rejecteds,
                    args=args, save_path=fw, run_name=run_name, accelerator=accelerator, sample_index=batch_idx
                )

                # Collect results for each sample in the batch
                for i, (prompt, result) in enumerate(zip(batch_prompts, res_iea)):
                    # print('result',result)
                    data = {
                        'prompt': str(prompt),
                        'chosen_given': result['chosen_given'],
                        'chosen_given_reward_score': result['chosen_given_reward_score'],
                        'rejected_given': result['rejected_given'],
                        'rejected_given_reward_score': result['rejected_given_reward_score'],
                        'original_response': result['original_output'],
                        'original_response_reward_score': result['original_output_reward_score'],
                        'optimized_response': result['text'],
                        'optimized_soft_reward_score': result['soft_reward_score'],
                        'optimized_hard_reward_score': result['hard_reward_score'],
                        'revised_optimized_response': result['revised_text'],
                        'revised_reward_score': result['revised_reward_score'],
                        'best_iter': result['best_iter'],
                        'time': elapsed_time
                    }
                    all_data.append(data)
            
            elif args.mode == 'bon' or args.mode == 'ori':
                res_bon, elapsed_time = bon(
                    model=model, tokenizer=tokenizer, reward_model=reward_model,reward_tokenizer=reward_tokenizer, 
                    device=device, prompt=batch_prompts, chosen=batch_chosens, rejected=batch_rejecteds,
                    args=args, save_path=fw, run_name=run_name, accelerator=accelerator
                )

                # Collect results for each sample in the batch
                for i, (prompt, result) in enumerate(zip(batch_prompts, res_bon)):
                    # print('result',result)
                    data = {
                        'prompt': str(prompt),
                        'best_bon_response': result['best_bon_response'],
                        'best_reward_score': result['best_reward_score'],
                        'all_reward_score': result['all_reward_score'],
                        'time': elapsed_time
                    }
                    all_data.append(data)

            elif args.mode == 'args':
                res_args, elapsed_time = args_1(
                    model=model, tokenizer=tokenizer, reward_model=reward_model,reward_tokenizer=reward_tokenizer, 
                    device=device, prompt=batch_prompts, chosen=batch_chosens, rejected=batch_rejecteds,
                    args=args, save_path=fw, run_name=run_name, accelerator=accelerator
                )
                for i, (prompt, result) in enumerate(zip(batch_prompts, res_args)):
                    # print('result',result)
                    data = {
                        'prompt': str(prompt),
                        'args_response': result['response'],
                        'reward_score': result['reward_score'],
                        'time': elapsed_time
                    }
                    all_data.append(data)

            elif args.mode == 'cbs':
                res_cbs, elapsed_time = cbs(
                    model=model, tokenizer=tokenizer, reward_model=reward_model,reward_tokenizer=reward_tokenizer, 
                    device=device, prompt=batch_prompts, chosen=batch_chosens, rejected=batch_rejecteds,
                    args=args, save_path=fw, run_name=run_name, accelerator=accelerator
                )
                for i, (prompt, result) in enumerate(zip(batch_prompts, res_cbs)):
                    # print('result',result)
                    data = {
                        'prompt': str(prompt),
                        'CBS_response': result['response'],
                        'reward_score': result['reward_score'],
                        'time': elapsed_time
                    }
                    all_data.append(data)

            if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
                with open(os.path.join(fw, args.save_file_name), 'w') as f:
                    json.dump(all_data, f, indent=4)


if __name__ == "__main__":
    main()
