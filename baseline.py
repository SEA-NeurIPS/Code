import gc
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import wandb
from collections import defaultdict

from accelerate.utils import gather_object
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import StoppingCriteria, StoppingCriteriaList
from util import *
from prompt import *
from model import *

import time
import seaborn as sns
import copy

from matplotlib import pyplot as plt
import logging
logger = logging.getLogger(__name__)


def bon(model, tokenizer, reward_model=None, reward_tokenizer=None, device='cuda', prompt="", chosen=None, rejected=None, args=None, save_path=None, run_name=None, accelerator=None):

    if (args.not_do_sample) and (args.n_of_bon > 1):
        raise

    start_time = time.time()
    num_sample = len(prompt)
    # tokenizer.padding_side = 'right'
    reward_tokenizer.padding_side = 'right'

    ### prepare model & mapping
    model.eval()
    reward_model.eval()

    if args.dataset == 'math':
        x = apply_template(args.pretrained_model, prompt, tokenizer, inser_sys=SYS_MESSAGE_MATH, add_generation_prompt=True)
    elif args.dataset == 'tqa':
        x = apply_template(args.pretrained_model, prompt, tokenizer, inser_sys=SYS_MESSAGE_TQA, add_generation_prompt=True)
    elif args.dataset == 'advbench' and args.jail_break:
        x = apply_template(args.pretrained_model, prompt, tokenizer, add_generation_prompt=True)
        attack_prompt = copy.deepcopy(prompt)
        for i in range(len(prompt)):
            attack_prompt[i].append({"role":"assistant", "content":JAIL_BREAK_DICT[args.jail_break]})
        x = apply_template(args.pretrained_model, attack_prompt, tokenizer, add_generation_prompt=False)
        for i in range(len(x)):
            x[i]=x[i].replace("<|eot_id|>",'') # may be different symbol for different model
    else:
        x = apply_template(args.pretrained_model, prompt, tokenizer, add_generation_prompt=True)
    
    if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
        logging.info("[prompt]: %s" % (x))

    ### stop if encouter some ids
    stop_ids = [tokenizer.eos_token_id]
    if 'llama-3' in args.pretrained_model:
        stop_ids.extend([128001,128009]) # template ids of llama-3
    stop_criteria = StoppingCriteriaList([StopOnSpecificTokens(stop_ids)])#.to(device)

    ### test generation from hard input
    model_inputs = tokenizer(x, return_tensors="pt", padding='longest')
    with torch.no_grad():
        generate_ids = model.generate(
            **model_inputs.to(device), 
            do_sample=False if args.not_do_sample else True,
            max_new_tokens=args.max_length, 
            num_return_sequences=args.n_of_bon,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            stopping_criteria=stop_criteria) #stopping_criteria=stop_criteria
    pure_generate_ids = generate_ids[:, model_inputs['input_ids'].shape[1]:]
    original_hard_output = tokenizer.batch_decode(pure_generate_ids, skip_special_tokens=True)

    del model_inputs, generate_ids, pure_generate_ids
    gc.collect()

    _prompt=[]
    for p in prompt:
        p=p*args.n_of_bon
        _prompt.extend(p)
    
    _original_hard_output = [[_prompt[i]] + [{"role":'assistant','content':original_hard_output[i]}] for i in range(len(_prompt))]
    if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
        logging.info("[response]: %s" % ([_original_hard_output[0]]))

    original_hard_output_rm = apply_template(args.reward_model, _original_hard_output, reward_tokenizer, add_generation_prompt=False)
    if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
        logging.info("[reward prompt]: %s" % ([original_hard_output_rm[0]]))
    
    with torch.no_grad():
        kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
        original_hard_output_reward_score = reward_model(**reward_tokenizer(original_hard_output_rm, **kwargs).to(device)).logits
    if args.reward_model=='RM-mistral-7b-dpa':
        original_hard_output_reward_score = original_hard_output_reward_score.squeeze()[DPA_DIM].tolist()
    else:
        original_hard_output_reward_score = original_hard_output_reward_score.squeeze(1).tolist()
    
    _original_hard_output_reward_score=[]
    for i,j in zip(original_hard_output_rm, original_hard_output_reward_score):
        _original_hard_output_reward_score.append((i,j))
        
    question_rewards = defaultdict(list)
    for key, reward in _original_hard_output_reward_score:
        question, response = key.split('assistant')[0], key 
        question_rewards[question].append((response, reward)) 

    best_responses = []
    for question, responses_rewards in question_rewards.items():
        
        best_response, best_reward = max(responses_rewards, key=lambda x: x[1])
        all_rewards = [reward for _, reward in responses_rewards]

        # Append the best response and its reward score
        best_responses.append({
            'best_bon_response': best_response,
            'all_reward_score': all_rewards,
            'best_reward_score': best_reward
        })
    end_time = time.time()
    elapsed_time = end_time - start_time

    return best_responses, elapsed_time


def _create_temp_sequences(
            sequence,
            topk_tokens,
            is_first_step=True,
            args=None
    ):
        """Create temporary sequences for reward computation."""
        if is_first_step:
            return topk_tokens.indices.view(-1, 1)
        return torch.cat(
            (sequence.repeat(args.top_k, 1), topk_tokens.indices.view(-1, 1)),
            dim=1
        )

def args_1(model, tokenizer, reward_model=None, reward_tokenizer=None, device='cuda', prompt="", chosen=None, rejected=None, args=None, save_path=None, run_name=None, accelerator=None):
    start_time = time.time()
    num_sample = len(prompt)
    reward_tokenizer.padding_side = 'right'

    ### prepare model & mapping
    model.eval()
    reward_model.eval()

    if args.dataset == 'math':
        x = apply_template(args.pretrained_model, prompt, tokenizer, inser_sys=SYS_MESSAGE_MATH, add_generation_prompt=True)
    elif args.dataset == 'tqa':
        x = apply_template(args.pretrained_model, prompt, tokenizer, inser_sys=SYS_MESSAGE_TQA, add_generation_prompt=True)
    elif args.dataset == 'advbench' and args.jail_break:
        x = apply_template(args.pretrained_model, prompt, tokenizer, add_generation_prompt=True)
        attack_prompt = copy.deepcopy(prompt)
        for i in range(len(prompt)):
            attack_prompt[i].append({"role":"assistant", "content":JAIL_BREAK_DICT[args.jail_break]})
        x = apply_template(args.pretrained_model, attack_prompt, tokenizer, add_generation_prompt=False)
        for i in range(len(x)):
            x[i]=x[i].replace("<|eot_id|>",'') # may be different symbol for different model
    else:
        x = apply_template(args.pretrained_model, prompt, tokenizer, add_generation_prompt=True)

    if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
        logging.info("[prompt]: %s" % (x))

    ### stop if encouter some ids
    stop_ids = [tokenizer.eos_token_id]

    ### test generation from hard input
    model_inputs = tokenizer(x, return_tensors="pt", padding='longest').input_ids.to(device)
    sequence = torch.empty(
            (1, 0),
            dtype=torch.int64,
            device=device
        )
    final_score = None
    for t in range(args.max_length):
        # Generate next token
        current_input = model_inputs if t == 0 else torch.cat(
            (model_inputs, sequence),
            dim=1
        )

        # Get generation output
        generate_kwargs = {
            'max_new_tokens': 1,
            'output_scores': True,
            'return_dict_in_generate': True,
            'renormalize_logits': True,
            'return_legacy_cache': True,
            'pad_token_id': tokenizer.eos_token_id,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "temperature":args.temperature
        }
        output = model.generate(inputs=current_input, **generate_kwargs)

        # Get top-k tokens and their probabilities
        topk_tokens = torch.topk(output["scores"][0][0], args.top_k)
        token_probs = topk_tokens.values

        # Create temporary sequences and compute rewards
        temp_sequences = _create_temp_sequences(sequence, topk_tokens, t == 0, args)
        # print('temp_sequences',temp_sequences)
        current_hard_output = tokenizer.batch_decode(temp_sequences, skip_special_token=True)
        _prompt=[]
        for p in prompt:
            p = p * args.top_k
            _prompt.extend(p)

        _current_hard_output = [[_prompt[i]] + [{"role":'assistant','content':current_hard_output[i]}] for i in range(len(_prompt))]
        current_hard_output_rm_all = apply_template(args.reward_model, _current_hard_output, reward_tokenizer, add_generation_prompt=False)
       
        with torch.no_grad():
            kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
            reward_scores = reward_model(**reward_tokenizer(current_hard_output_rm_all, **kwargs).to(device)).logits
            if args.reward_model == 'RM-mistral-b-dpa':
                    reward_scores = reward_scores.squeeze()[DPA_DIM].tolist()
            else:
                reward_scores = reward_scores.squeeze(1)

        score_tensor = token_probs + args.args_weight * reward_scores
        
        # Sample
        if args.args_mode == 'greedy':
            sampled_id = torch.argmax(score_tensor).item()
        else:
            sampled_id = torch.distributions.Categorical(logits=score_tensor).sample().item()
        sampled_token = topk_tokens.indices[sampled_id].view(1, 1)

        # Update sequence
        sequence = torch.cat((sequence, sampled_token), dim=1)
        # final_score = current_reward
        # Check for EOS token
        if sequence[0, -1].item() == tokenizer.eos_token_id:
            print(f"EOS BREAK at step {t}")            
            break
    end_time = time.time()
    elapsed_time = end_time - start_time
    hard_output = tokenizer.batch_decode(sequence, skip_special_token=True)
    output_all = [[_prompt[0]] + [{"role":'assistant','content':hard_output}]]
    rm_output_template = apply_template(args.reward_model, output_all, reward_tokenizer, add_generation_prompt=False)
    final_score = reward_model(**reward_tokenizer(rm_output_template[0], **kwargs).to(device)).logits.tolist()
    ret = []
    ret.append({'response': hard_output[0],
                'reward_score': final_score[0]
                })
    #print(ret)
    return ret, elapsed_time



def cbs(model, tokenizer, reward_model=None, reward_tokenizer=None, device='cuda', prompt="", chosen=None, rejected=None, args=None, save_path=None, run_name=None, accelerator=None, **model_kwargs):
    
    start_time = time.time()
    reward_tokenizer.padding_side = 'right'

    ### prepare model & mapping
    model.eval()
    reward_model.eval()

    w = args.beam_width
    k = args.successors_per_state
    l = args.chunk_length
    tokens_remain_per_chunk = l
    
    ### test model template
    if args.dataset == 'math':
        x = apply_template(args.pretrained_model, prompt, tokenizer, inser_sys=SYS_MESSAGE_MATH, add_generation_prompt=True)
    elif args.dataset == 'tqa':
        x = apply_template(args.pretrained_model, prompt, tokenizer, inser_sys=SYS_MESSAGE_TQA, add_generation_prompt=True)
    elif args.dataset == 'advbench' and args.jail_break:
        x = apply_template(args.pretrained_model, prompt, tokenizer, add_generation_prompt=True)
        attack_prompt = copy.deepcopy(prompt)
        for i in range(len(prompt)):
            attack_prompt[i].append({"role":"assistant", "content":JAIL_BREAK_DICT[args.jail_break]})
        x = apply_template(args.pretrained_model, attack_prompt, tokenizer, add_generation_prompt=False)
        for i in range(len(x)):
            x[i]=x[i].replace("<|eot_id|>",'') # may be different symbol for different model
    else:
        x = apply_template(args.pretrained_model, prompt, tokenizer, add_generation_prompt=True)

    if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
        logging.info("[prompt]: %s" % (x))
    logits_wrapper = LogitsProcessorList()
    if args.temperature:
        logits_wrapper.append(TemperatureLogitsWarper(args.temperature))
    if args.top_k:
        logits_wrapper.append(TopKLogitsWarper(args.top_k))
    if args.top_p:
        logits_wrapper.append(TopPLogitsWarper(args.top_p))
    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(MaxLengthCriteria(max_length=args.max_length))
    stop_ids = [tokenizer.eos_token_id]
    if 'llama-3' in args.pretrained_model:
        stop_ids.extend([128001,128009]) # template ids of llama-3
    stopping_criteria.append(StopOnSpecificTokens(stop_ids))
    has_eos_stopping_criteria = any(hasattr(criteria, "stop_ids") for criteria in stopping_criteria)
    if tokenizer.pad_token_id is not None:
        if 'llama-3' in args.pretrained_model:
            pad_token_id = 128001
        else:
            pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = tokenizer.pad_token_id

    
    inputs = tokenizer(x, return_tensors="pt", padding='longest')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    input_ids = input_ids.expand(w * k, -1)
    model_kwargs["attention_mask"] = attention_mask.expand(w * k, -1)
    prompt_len = input_ids.size(1)

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False
    prompt_len = input_ids.size(1)
    model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)
    while not this_peer_finished:
        model_inputs = model.prepare_inputs_for_generation(input_ids)
        outputs = model(
            **model_inputs,
            return_dict=True,
        )

        next_token_logits = outputs.logits[:, -1, :]

        next_token_logits = logits_wrapper(input_ids, next_token_logits)

        probs = nn.functional.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        tokens_remain_per_chunk -= 1
        
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
        this_peer_finished = unfinished_sequences.max() == 0

        if tokens_remain_per_chunk <= 0 or this_peer_finished:
            tokens_remain_per_chunk = l
            responses = tokenizer.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True)
            _prompt=[]
            for p in prompt:
                p=p * (w * k)
                _prompt.extend(p)

            _original_hard_output = [[_prompt[i]] + [{"role":'assistant','content':responses[i]}] for i in range(len(_prompt))]
            if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
                logging.info("[response]: %s" % ([_original_hard_output[0]]))
                
            current_output_rm = apply_template(args.reward_model, _original_hard_output, reward_tokenizer, add_generation_prompt=False)
            if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
                logging.info("[reward prompt]: %s" % ([current_output_rm[0]]))
    
            with torch.no_grad():
                kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
                original_hard_output_reward_score = reward_model(**reward_tokenizer(current_output_rm, **kwargs).to(device)).logits
                if args.reward_model=='RM-mistral-7b-dpa':
                    reward_scores = original_hard_output_reward_score.squeeze()[DPA_DIM]
                else:
                    reward_scores = original_hard_output_reward_score.squeeze(1)
            _, beam_idx = torch.topk(reward_scores, w, dim=0, largest=True, sorted=True)
            beam_idx = beam_idx.squeeze().repeat(k)
            input_ids = input_ids[beam_idx]
            unfinished_sequences = unfinished_sequences[beam_idx]
                
    response = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
    
    output = [prompt[0] + [{"role":'assistant','content':response}]]
    rm_output_template = apply_template(args.reward_model, output, reward_tokenizer, add_generation_prompt=False)
    with torch.no_grad():
        kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
        final_score = reward_model(**reward_tokenizer(rm_output_template, **kwargs).to(device)).logits
        if args.reward_model=='RM-mistral-7b-dpa':
            final_score = final_score.squeeze()[DPA_DIM].tolist()
        else:
            final_score = final_score.squeeze(1).tolist()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    ret = []
    ret.append({'response': response,
               'reward_score': final_score[0]
                })
    return ret, elapsed_time