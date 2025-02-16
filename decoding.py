import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import wandb

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

BIG_CONST = 1e10

def decode(model, tokenizer, reward_model=None, reward_tokenizer=None, device='cuda', prompt="", chosen=None, rejected=None, args=None, save_path=None, run_name=None, accelerator=None, sample_index=None):

    same_source = if_same_tokenizer(args.reward_model, args.pretrained_model)
    num_sample = len(prompt)

    dim = DPA_DIM if args.reward_model=='RM-armorm-llama-3-8b' else None
    print('dim',dim)

    if num_sample > 1: 
        raise ValueError("Prompt batch is not supported due to reward model cannot handle batch-wise padding properly with soft input. Will be fixed later.")
    
    base_vacab_size = len(tokenizer) if 'pythia' not in args.pretrained_model else model.get_input_embeddings().weight.shape[0]
    reward_vacab_size = len(reward_tokenizer)

    if args.wandb and ((not args.accelerator) or (args.accelerator and accelerator.is_main_process)):
        wandb.init(project='EBoN', config=args, name="{}_{}".format(args.dataset, run_name))

    ### prepare model & mapping
    model.eval()
    reward_model.eval()
    mapping = get_tokenizer_mapping(tokenizer, reward_tokenizer, base_vacab_size, reward_vacab_size, device, same_source = same_source)

    ### test model template
    if args.dataset == 'math':
        x = apply_template(args.pretrained_model, prompt, tokenizer, inser_sys=SYS_MESSAGE_MATH, add_generation_prompt=True)
    elif args.dataset == 'tqa':
        x = apply_template(args.pretrained_model, prompt, tokenizer, inser_sys=SYS_MESSAGE_TQA, add_generation_prompt=True)
    elif args.dataset == 'advbench' and args.jail_break:
        x = apply_template(args.pretrained_model, prompt, tokenizer, add_generation_prompt=True)
        attack_prompt = copy.deepcopy(prompt)
        for i in range(len(prompt)):
            attack_prompt[i].append({"role":"assistant", "content":JAIL_BREAK_DICT[args.jail_break]}) #Sure, i can assist you with it.
        x = apply_template(args.pretrained_model, attack_prompt, tokenizer, add_generation_prompt=False)
        for i in range(len(x)):
            x[i]=x[i].replace("<|eot_id|>",'') # may be different symbol for different model
    else:
        x = apply_template(args.pretrained_model, prompt, tokenizer, add_generation_prompt=True)

    if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
        logging.info("[Model Prompt]\n%s" % (x))
    x_ = tokenizer(x, return_tensors="pt", padding='longest').to(device)
    x_t = x_["input_ids"]
    x_m = x_["attention_mask"]
    x_onehot = one_hot(x_t, dimension=base_vacab_size) 
    
    chosen_given_reward_score = [None] * num_sample
    rejected_given_reward_score = [None] * num_sample
    original_hard_output_reward_score = [None] * num_sample
    original_hard_output = [None] * num_sample

    ### stop if encouter some ids
    stop_ids = [tokenizer.eos_token_id]
    if 'llama-3' in args.pretrained_model:
        stop_ids.extend([128001, 128009, 128006]) # template ids of llama-3
    stop_criteria = StoppingCriteriaList([StopOnSpecificTokens(stop_ids)])

    if args.verify:
        ### test reward of the chosen given by dataset
        if (len(chosen) > 0) and (None not in chosen):
            chosen_given_reward_score = compute_reward_score(args, reward_tokenizer, reward_model, chosen, device=device).squeeze(1).tolist()
            # chosen_given_rm = apply_template(args.reward_model, chosen, reward_tokenizer, add_generation_prompt=False)
            # with torch.no_grad():
            #     kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
            #     if args.reward_model=='RM-armorm-llama-3-8b':
            #         chosen_given_reward_score = reward_model(**reward_tokenizer(chosen_given_rm, **kwargs).to(device)).rewards[:,DPA_DIM]
            #         print('chosen_given_reward_score',chosen_given_reward_score)
            #         exit(0)
            #     else:
            #         chosen_given_reward_score = reward_model(**reward_tokenizer(chosen_given_rm, **kwargs).to(device)).logits
            # if args.reward_model=='RM-mistral-7b-dpa':
            #     chosen_given_reward_score = chosen_given_reward_score.squeeze()[DPA_DIM].tolist()
            # else:
            #     chosen_given_reward_score = chosen_given_reward_score.squeeze(1).tolist()
            if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
                logging.info("[Given Chosen] Reward: %s\n%s" % (chosen_given_reward_score, [i[-1]["content"] for i in chosen]))

        ### test reward of the rejected given by dataset
        if (len(rejected)) > 0 and (None not in rejected):
            rejected_given_reward_score = compute_reward_score(args, reward_tokenizer, reward_model, rejected, device=device).squeeze(1).tolist()
            # rejected_given_rm = apply_template(args.reward_model, rejected, reward_tokenizer, add_generation_prompt=False)
            # with torch.no_grad():
            #     kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
            #     rejected_given_reward_score = reward_model(**reward_tokenizer(rejected_given_rm, **kwargs).to(device)).logits
            # if args.reward_model=='RM-mistral-7b-dpa':
            #     rejected_given_reward_score = rejected_given_reward_score.squeeze()[DPA_DIM].tolist()
            # else:
            #     rejected_given_reward_score = rejected_given_reward_score.squeeze(1).tolist()
            if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
                logging.info("[Given Rejected] Reward: %s\n%s" % (rejected_given_reward_score, [i[-1]["content"] for i in rejected]))

        ### test generation from hard input
        model_inputs = tokenizer(x, return_tensors="pt", padding='longest')
        with torch.no_grad():
            generate_ids = model.generate(**model_inputs.to(device), do_sample=False, max_new_tokens=args.max_length, stopping_criteria=stop_criteria) #max_length=args.length + model_inputs.shape[-1]
            if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
                logging.info("[Original Hard Output Full]\n%s" % (tokenizer.batch_decode(generate_ids, skip_special_tokens=False)))
        
        pure_generate_ids = generate_ids[:, model_inputs['input_ids'].shape[1]:]
        hard_non_padding_lengths = (pure_generate_ids != tokenizer.pad_token_id).sum(dim=1)
        # if args.length == 0:
        #     length = max(non_padding_lengths)
        #     if length > args.max_length:
        #         length = args.max_length
        #     if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
        #         logging.info("Changing generation length to: {}".format(length))
        # else:
        #     length = args.length
        original_hard_output = tokenizer.batch_decode(pure_generate_ids, skip_special_tokens=True)

        ### test reward of the generation from hard input
        _original_hard_output = [prompt[i] + [{"role":'assistant','content':original_hard_output[i]}] for i in range(len(prompt))]
        original_hard_output_reward_score = compute_reward_score(args, reward_tokenizer, reward_model, _original_hard_output, device=device).squeeze(1).tolist()
        # original_hard_output_rm = apply_template(args.reward_model, _original_hard_output, reward_tokenizer, add_generation_prompt=False)
        # with torch.no_grad():
        #     kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
        #     original_hard_output_reward_score = reward_model(**reward_tokenizer(original_hard_output_rm, **kwargs).to(device)).logits
        # if args.reward_model=='RM-mistral-7b-dpa':
        #     original_hard_output_reward_score = original_hard_output_reward_score.squeeze()[DPA_DIM].tolist()
        # else:
        #     original_hard_output_reward_score = original_hard_output_reward_score.squeeze(1).tolist()
        if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
            logging.info("[Original Hard Output] Reward: %s\n%s" % (original_hard_output_reward_score, original_hard_output))

        ### test generation from soft input
        with torch.no_grad():
            x_model_outputs = model(input_ids=x_t[:,:-1], attention_mask=x_m[:,:-1], use_cache=True)
        x_model_past = x_model_outputs.past_key_values

        text, _, _ = decode_with_model_topk_x(model=model, x_onehot=x_onehot, topk=args.topk, x_past=x_model_past, tokenizer=tokenizer, length=max(hard_non_padding_lengths))
        if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
            logging.info("[Original Soft Output]\n%s" % (text))

        ### test if tokenizer of base model & reward model match
        rm_onehot_x = mapping_tokenizer(x_onehot, mapping, same_source=same_source, device=device)
        text, _, _ = get_text_from_logits(x_onehot.float(), tokenizer)
        rm_text, _, _ = get_text_from_logits(rm_onehot_x.float(), reward_tokenizer)
        if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
            logging.info("[Model Decoding]\n%s" % (text))
            logging.info("[Mapping to Reward Model Decoding]\n%s" % (rm_text))

    ### test reward model template
    rm_x = apply_template(args.reward_model, prompt, reward_tokenizer, add_generation_prompt=True)
    if args.reward_model=='RM-armorm-llama-3-8b': # RM-armorm-llama-3-8b will not add generation prompt, even add_generation_prompt is set True, so we manully do it.
        rm_x = [i + "<|start_header_id|>assistant<|end_header_id|>\n\n" for i in rm_x]
    if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
        logging.info("[Reward Model Prompt]\n%s" % (rm_x))

    ### prepare x & repeat batch_size times
    x_t = x_t.repeat_interleave(repeats=args.batch_size, dim=0)
    x_m = x_m.repeat_interleave(repeats=args.batch_size, dim=0)
    x_onehot = x_onehot.repeat_interleave(repeats=args.batch_size, dim=0)
    soft_forward_x = x_onehot[:, -1:, :] 
    with torch.no_grad():
        x_model_outputs = model(input_ids=x_t[:,:-1], attention_mask=x_m[:,:-1], use_cache=True)
    x_model_past = x_model_outputs.past_key_values

    rm_x_ = reward_tokenizer(rm_x, return_tensors="pt", padding='longest').to(device)
    rm_x_t = rm_x_["input_ids"]
    rm_x_m = rm_x_["attention_mask"]
    rm_x_onehot = one_hot(rm_x_t, dimension=len(reward_tokenizer)) 
    rm_x_t = rm_x_t.repeat_interleave(repeats=args.batch_size, dim=0)
    rm_x_m = rm_x_m.repeat_interleave(repeats=args.batch_size, dim=0)
    rm_x_onehot = rm_x_onehot.repeat_interleave(repeats=args.batch_size, dim=0)

    if args.reward_model=='RM-armorm-llama-3-8b':
        rm_soft_forward_x = rm_x_onehot
        rm_x_model_past = None
    else:
        rm_soft_forward_x = rm_x_onehot[:, -1:, :] 
        with torch.no_grad():
            rm_x_model_outputs = reward_model(input_ids=rm_x_t[:,:-1], attention_mask=rm_x_m[:,:-1], use_cache=True)
        rm_x_model_past = rm_x_model_outputs.past_key_values


    ### prepare z for reward model & repeat batch_size times
    if 'llama-3' in args.reward_model or args.reward_model == 'RM-mistral-7b':
        
        if args.reward_model == 'RM-llama-3-8b':
            rm_z_t = torch.Tensor([128009, 128006, 78191, 128007, 271]).repeat(num_sample, 1).to(torch.int64).to(device)
            rm_z_m = torch.Tensor([1, 1, 1, 1, 1]).repeat(num_sample, 1).to(device)
        elif (args.reward_model == 'RM-fsfairx-llama-3-8b') or (args.reward_model == 'RM-llama-3.2-3b') or  (args.reward_model == 'RM-armorm-llama-3-8b'):
            rm_z_t = torch.Tensor([128009]).repeat(num_sample, 1).to(torch.int64).to(device)
            rm_z_m = torch.Tensor([1]).repeat(num_sample, 1).to(device)
        elif args.reward_model == 'RM-mistral-7b':
            rm_z_t = torch.Tensor([2]).repeat(num_sample, 1).to(torch.int64).to(device)
            rm_z_m = torch.Tensor([1]).repeat(num_sample, 1).to(device)

        rm_z_onehot = one_hot(rm_z_t, dimension=len(reward_tokenizer))

        rm_z_t = rm_z_t.repeat_interleave(repeats=args.batch_size, dim=0)
        rm_z_m = rm_z_m.repeat_interleave(repeats=args.batch_size, dim=0)
        rm_z_onehot = rm_z_onehot.repeat_interleave(repeats=args.batch_size, dim=0)
    else:
        rm_z_onehot=None

    ### prepare z for  base model
    if  'llama-3' in args.pretrained_model: 
        z_t = torch.Tensor([128009]).repeat(num_sample, 1).to(torch.int64).to(device)
        z_m = torch.Tensor([1]).repeat(num_sample, 1).to(device)
    else:
        z = [tokenizer.eos_token] * num_sample
        z_ = tokenizer(z, return_tensors="pt", padding='longest', add_special_tokens=False).to(device)
        z_t = z_["input_ids"]
        z_m = z_["attention_mask"]
    z_onehot = one_hot(z_t, dimension=base_vacab_size)
    z_t = z_t.repeat_interleave(repeats=args.batch_size, dim=0)
    z_m = z_m.repeat_interleave(repeats=args.batch_size, dim=0)
    z_onehot = z_onehot.repeat_interleave(repeats=args.batch_size, dim=0)

    ### init y & optim & scheduler
    if args.init_mode == 'original':
        init_logits = initialize(model, x_t, x_m, args.init_temp, args.batch_size ,device, tokenizer, args.pretrained_model, length=None)
    elif args.init_mode == 'prompt':
        if args.dataset == 'gsm8k':
            x_better = apply_template(args.pretrained_model, prompt, tokenizer, inser_sys=SYS_MESSAGE_GSM8K, add_generation_prompt=True)           
        else:
            x_better = apply_template(args.pretrained_model, prompt, tokenizer, inser_sys=SYS_MESSAGE, add_generation_prompt=True)
        x_better = tokenizer(x_better, return_tensors="pt", padding='longest').to(device)   
        init_logits = initialize(model, x_better.input_ids, x_better.attention_mask, args.init_temp, args.batch_size ,device, tokenizer, args.pretrained_model, length=None)
    elif args.init_mode == 'bon':
        pass
    elif args.init_mode == 'random':
        if args.length == 0:
            length = max(hard_non_padding_lengths)
        else:
            length = args.length
        init_logits =  torch.randn([args.batch_size, length, base_vacab_size], device=device)
    else:
        raise

    if args.length == 0:
        length = init_logits.shape[1]
        if length > args.max_length:
            length = args.max_length
            init_logits = init_logits[:,:length,:]
        
        if length < args.min_length:
            length = args.min_length
            init_logits = torch.cat([init_logits, torch.randn([args.batch_size, length - init_logits.shape[1], base_vacab_size], device=device)], dim=1)
        
        if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
            logging.info("Changing generation length to: {}".format(length))
    else:
        if args.length < init_logits.shape[1]:
            length = args.length
            init_logits = init_logits[:,:length,:]
        else:
            length = args.length
            init_logits = torch.cat([init_logits, torch.randn([args.batch_size, length - init_logits.shape[1], base_vacab_size], device=device)], dim=1)

    text, _, _ = get_text_from_logits(init_logits, tokenizer)
    if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
        for bi in range(num_sample):
            logging.info("[initial]: %s" % ([text[bi*args.batch_size:(bi+1)*args.batch_size]]))
    
    y_logits = init_logits
    y_logits_ = None
    noise_std = 0.0
    noise = 0
    
    epsilon = torch.nn.Parameter(y_logits.clone())
    # nn.init.normal_(epsilon)
    if args.optim == 'adam':
        optim = torch.optim.Adam([epsilon], lr=args.lr)
    elif args.optim == 'sgd':
        optim = torch.optim.SGD([epsilon], lr=args.lr, momentum=0.9)
    else:
        raise
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=args.scheduler_stepsize,gamma=args.scheduler_gamma)
    frozen_len = args.frozen_length

    # if args.accelerator:
    #     epsilon, optim = accelerator.prepare(epsilon, optim)

    ### best logging & early stop
    stop_counter = 0 
    best_reward_score = torch.full((args.batch_size * num_sample,),-1e10,device=device)  # Shape: [batch_size * outter_batch_size]
    best_iter = torch.full((args.batch_size * num_sample,),-1,device=device)  # Shape: [batch_size * outter_batch_size]
    best_text = [None] * args.batch_size * num_sample

    ### some masks
    x_mask = None
    mask_t = None
    rm_mask_t = None
    
    if 'llama-3' in args.pretrained_model:
        bad_id = [128009, 128007,128006, 128001, 128000, 128037, 128181, 128183]
    elif args.pretrained_model == 'llama-2-7b-base':
        bad_id = [1,518,25580,29962,29914]
    elif args.pretrained_model == 'mistral-7b-base' or args.pretrained_model == 'mistral-7b-instruct':
        bad_id = [1,2]
    elif 'pythia' in args.pretrained_model:
        bad_id = [0] #535
    else:
        bad_id = []

    if len(bad_id) > 0:
        bad_mask = torch.ones([base_vacab_size]).to(device)
        bad_mask[bad_id] = 0
        bad_mask = bad_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size * num_sample, length, 1)
    else:
        bad_mask = None

    if args.token_kl:
        per_token_kl_list = []

    '''
    Begin SGLD
    '''
    elapsed_time = 0
    for iter in range(args.num_iters):
        
        start_time = time.time()

        optim.zero_grad()

        # epsilon = top_k_filter_3d(epsilon, 0, mask=None, extra_mask=None, bad_mask=bad_mask)
        y_logits_ = epsilon + noise # + y_logits 
        # y_logits_t0 = y_logits_
        y_logits_ = top_k_filter_3d(y_logits_, 0, mask=None, extra_mask=None, bad_mask=bad_mask)

        '''
        Forward into Base Model
        '''
        soft_forward_y = y_logits_ / 0.001
        if args.straight_through:
            if mask_t is None:
                soft_forward_y = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
            else:
                soft_forward_y = top_k_filter_3d(y_logits_, args.topk, mask=mask_t, extra_mask=x_mask, bad_mask=bad_mask) / 0.001

        if args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y_logits_t, _ = soft_forward(model, x_onehot=soft_forward_x, y_logits=soft_forward_y, z_onehot=None, x_past=x_model_past, detach_y = args.detach)
        else:
            y_logits_t, _ = soft_forward(model, x_onehot=soft_forward_x, y_logits=soft_forward_y, z_onehot=None, x_past=x_model_past, detach_y = args.detach)

        if args.topk == 0:
            mask_t = None
        else:
            _, indices_t = torch.topk(y_logits_t, args.topk)
            mask_t = torch.zeros_like(y_logits_t).scatter_(2, indices_t, 1)
        
        flu_loss = soft_nll( 
            pi = y_logits_ / args.input_lgt_temp,
            ref = y_logits_t / args.output_lgt_temp,
            k = args.topk,
            div = args.div,
            bad_mask = bad_mask
            )


        end_loss = torch.zeros((args.batch_size * num_sample, 1), device=device)
        '''
        # Make sure it stops
        soft_forward_y_ = y_logits_ #(y_logits_.detach() / 0.1 - y_logits_).detach() + y_logits_ #
        if args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                xyz_logits, xy_length = soft_forward_xyz(model, soft_forward_x, soft_forward_y_, z_onehot)
        else:
            xyz_logits, xy_length = soft_forward_xyz(model, soft_forward_x, soft_forward_y_, z_onehot)
        # Reshaping
        bz = args.batch_size * num_sample
        lg = xyz_logits.shape[1]
        st = xy_length - 1
        ed = xyz_logits.shape[1] - 1
        xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
        z_logits = torch.cat([xyz_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)
        end_loss = torch.nn.CrossEntropyLoss(reduction='none')(
            z_logits,
            z_t.view(-1))
        end_loss = end_loss.view(args.batch_size * num_sample, -1).mean(-1)
        '''


        '''
        Forward into Reward Model
        '''
        rm_soft_forward_y = y_logits_ / args.reward_temp
        rm_soft_forward_y = mapping_tokenizer(rm_soft_forward_y, mapping, same_source=same_source, device=device)

        if args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                reward_score = soft_forward(reward_model, x_onehot=rm_soft_forward_x, y_logits=rm_soft_forward_y, z_onehot=rm_z_onehot, x_past=rm_x_model_past, dim=dim)
        else:
            reward_score = soft_forward(reward_model, x_onehot=rm_soft_forward_x, y_logits=rm_soft_forward_y, z_onehot=rm_z_onehot, x_past=rm_x_model_past, dim=dim)
        if args.reward_model=='RM-mistral-7b-dpa':
            reward_score = reward_score[:, DPA_DIM].unsqueeze(1)

        ### compute Loss & backward
        loss = - args.reward_weight * reward_score +  args.flu_weight * flu_loss + args.end_weight * end_loss
        loss = loss.mean()
        if iter < args.num_iters - 1: 
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(epsilon, max_norm=1)
            optim.step()
            scheduler.step() 
        last_lr = scheduler.get_last_lr()[0]

        ### Adding Noise
        if iter < args.num_iters - 1:
            large_noise_iters = [int(_) for _ in args.large_noise_iters.split(',')]
            large_gs_stds = [float(_) for _ in args.large_gs_std.split(',')]
            noise_std = 0.
            if iter % args.noise_iters == 0:
                noise_last = True
                for ni in range(len(large_noise_iters)):
                    if iter < large_noise_iters[ni]:
                        noise_last = False
                        break
                if noise_last:
                    noise_std = args.gs_std
                else:
                    noise_std = large_gs_stds[ni]

                noise = torch.normal(mean=args.gs_mean, std=noise_std, size=epsilon.size(), device=device, requires_grad=False)
                if args.win_anneal_iters >= 0 and iter >= args.win_anneal_iters:
                    zeros = torch.zeros_like(noise)
                    noise_mix = torch.cat([zeros[:, :frozen_len], noise[:, frozen_len:]], dim=1)
                    # y_logits = y_logits + noise_mix
                    noise = noise_mix
                else:
                    # y_logits = y_logits + noise
                    noise = noise


        if args.token_kl:
            good_words = ['safe', 'safety', 'cannot', 'cant', 'sorry', 'not', 'unable', 'apologize', 'legal', 'but', 'however', 'ethical', 'law' ,'ethical', 'can\'t', 'against', 'violate', 'legal', 'laws', 'legally', 'unethical', 'unlawful', 'criminal', 'malicious', 'illegal','no','harmful', 'harm', 'bad', 'danger', 'dangerous', 'OpenAI', 'morality','impossible']
            # print([" "+i for i in good_words] )
            # exit(0)
            good_words = good_words + [i.upper() for i in good_words] + [i.lower() for i in good_words] + [i[0].upper()+i[1:] for i in good_words] 
            good_words = good_words + [" "+i for i in good_words] 
            # print('good_words',good_words)
            # exit(0)
            _good_index = [tokenizer.encode(i)[1:] for i in good_words]
            good_index = []
            for i in _good_index:
                good_index = good_index + i
            good_index=list(set(good_index))
            # print('aaa',tokenizer.decode(good_index[0]))
            # exit(0)
            k=10
            with torch.no_grad():
                #y_logits_mask = decode_with_model_topk(model, y_logits_, args.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=None, bad_mask=None, return_mask_logits=True)
                increase_word_list=[]
                decrease_word_list=[]
                
                per_token_kl, topk_increase_idx, topk_increase_diff, topk_decrease_idx, topk_decrease_diff = get_per_token_kl(init_logits, y_logits_, topk=k)
                
                good_word_count=[0]*topk_increase_idx.shape[1]

                for i in range(topk_increase_idx.shape[1]):
                    increase_ids = topk_increase_idx[:,i,:].detach().tolist()[0]
                    increase_values = topk_increase_diff[:,i,:].detach().tolist()[0]
                    per_position={}
                    for word, value in zip(increase_ids, increase_values):
                        
                        if word in good_index:
                            # print('word',word)
                            good_word_count[i]=good_word_count[i]+1
                        increase_word = tokenizer.decode(word)
                        # print('However', tokenizer.encode("However"))
                        # print(' However', tokenizer.encode(" However"))
                        # print('word',word)
                        # print('increase_word',increase_word)
                        per_position[increase_word]=value
                    # exit(0)
                    increase_word_list.append(per_position)

                    decrease_ids = topk_decrease_idx[:,i,:].detach().tolist()[0]
                    decrease_values = topk_decrease_diff[:,i,:].detach().tolist()[0]
                    per_position={}
                    for word, value in zip(decrease_ids, decrease_values):
                        decrease_word = tokenizer.decode(word)
                        per_position[decrease_word]=value
                    decrease_word_list.append(per_position)
                # plot_and_save_per_token_kl(per_token_kl, os.path.join(save_path, "per_token_kl_{}.png".format(iter)))
                per_token_kl_list.append(per_token_kl)

            output_data = {
                'increase_word_list': increase_word_list,
                'decrease_word_list': decrease_word_list
            }

            if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
                print('good_word_count',good_word_count)
                good_word_count_file = os.path.join(save_path, f"good_word_count_index{sample_index}_top{k}.txt")
                with open(good_word_count_file, 'a') as file:
                    file.write(f"{good_word_count}\n")
                with open(os.path.join(save_path, f"word_lists_{iter}_{k}.json"), 'w') as json_file:
                    json.dump(output_data, json_file, ensure_ascii=False, indent=4)

       
        with torch.no_grad():
            text, _, _ = get_text_from_logits(y_logits_, tokenizer)
            masked_text, _, _ = decode_with_model_topk(model, y_logits_, args.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=None, bad_mask=None)

        _output_texts=[]
        for i in range(num_sample):
            _output_texts.extend([prompt[i] +  [{"role":'assistant','content':j}] for j in  text[i*args.batch_size: (i+1) * args.batch_size]])
        for i in range(num_sample):
            _output_texts.extend([prompt[i] +  [{"role":'assistant','content':j}] for j in  masked_text[i*args.batch_size: (i+1) * args.batch_size]])

        #_output_texts = apply_template(args.reward_model, _output_texts, reward_tokenizer, add_generation_prompt=False)
        
        #with torch.no_grad():
        hard_reward_score = compute_reward_score(args, reward_tokenizer, reward_model, _output_texts, device=device)

            # kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
            # _output_texts = reward_tokenizer(_output_texts, **kwargs).to(device)
            # hard_reward_score = reward_model(**_output_texts).logits.squeeze(-1)

        hard_reward_score_nomask = hard_reward_score.squeeze(-1)[:len(text)]
        hard_reward_score_mask = hard_reward_score.squeeze(-1)[len(text):]

        ### verbose & log to wandb
        if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):

            if args.verbose and ((iter + 1) % args.print_every == 0 or iter == 0 or iter + 1 == args.num_iters):
                for bi in range(args.batch_size):
                    logging.info(
                        "%d, loss: %.4f, reward_score: %.4f, flu_loss: %.4f, end_loss: %.4f, hard_reward_nomask: %.4f, hard_reward_mask: %.4f,lr: %.4f" % (
                        iter + 1, loss.item(), reward_score[bi].item(), flu_loss[bi].item(), end_loss[bi].item(), hard_reward_score_nomask[bi].item(), hard_reward_score_mask[bi].item(), last_lr))
                logging.info(f"Decoded Texts\n{text}\n{masked_text}")

            if args.wandb:
                wandb.log({
                    "loss/loss": loss.item(),
                    "loss/reward_score": reward_score.mean().item(),
                    "loss/flu_loss": flu_loss.mean().item(),
                    "loss/end_loss": end_loss.mean().item(),
                    "loss/hard_reward_nomask": hard_reward_score_nomask.mean().item(),
                    "loss/hard_reward_mask": hard_reward_score_mask.mean().item(),
                    "grad/iteration": iter + 1,
                    "grad/learning_rate": scheduler.get_last_lr()[0],
                    "grad/grad_norm": epsilon.grad.norm() if epsilon.grad is not None else 0, 
                    "grad/grad_min": epsilon.grad.min() if epsilon.grad is not None else 0, 
                    "grad/grad_max": epsilon.grad.max() if epsilon.grad is not None else 0, 
                })

            if args.debug:
                debug(args, prompt, mapping, reward_model, model, reward_tokenizer, tokenizer, rm_soft_forward_x, rm_x_onehot, soft_forward_x, y_logits_, rm_x_model_past, x_model_past, device)
        
        '''
        Update Best Records
        '''
        with torch.no_grad():
            current_scores = hard_reward_score_mask.reshape(best_reward_score.shape).to(device)  # Shape: [batch_size * outter_batch_size]
            better_mask = current_scores.to(device) > best_reward_score.to(device)  # Shape: [batch_size * outter_batch_size]
            better_indices = torch.nonzero(better_mask, as_tuple=True)[0]

            if len(better_indices) > 0:
                # Update best_reward_score and best_iter tensors
                stop_counter = 0
                best_reward_score[better_mask] = current_scores[better_mask].float()
                best_iter[better_mask] = iter + 1  # Assuming iter starts at 0
                for idx in better_indices.tolist():
                    best_text[idx] = masked_text[idx]
                logging.info(f"Better Reward at iteration {best_iter.detach().tolist()}, best reward score: {best_reward_score.detach().tolist()}\n{text}\n{masked_text}")
            else:
                stop_counter = stop_counter + 1
            if  stop_counter > args.patience:
                logging.info("Break")
                break


    if args.token_kl:
        per_token_kl_tensor = torch.stack(per_token_kl_list, dim=0).cpu().detach() #torch.cat(per_token_kl_list, dim=0)  # 拼接所有迭代的 KL 散度，得到 [num_iters * batch_size, seq_len]
        torch.save(per_token_kl_tensor, os.path.join(save_path, "per_token_kl_tensor_index{}.pt".format(sample_index)))
        plot_and_save_per_token_kl_3d(per_token_kl_tensor, save_path=os.path.join(save_path, "per_token_kl_3d_idenx{}.png".format(sample_index)))

    
    best_text_revise=[]
    if args.revise_mode == 'A':
        best_text_revise = [[{"role":"user", "content": REVISE_PROMPT_NoQ.format(i)}] for i in best_text]
    elif args.revise_mode == 'QA':
        for i in range(num_sample):
            for j in  best_text[i*args.batch_size: (i+1) * args.batch_size]:
                best_text_revise.append([
                    {"role":"user",
                     "content": REVISE_PROMPT.format(prompt[i][-1]["content"], j) 
                     }])
    elif args.revise_mode == 'QA-ICL':
        for i in range(num_sample):
            for j in  best_text[i*args.batch_size: (i+1) * args.batch_size]:
                best_text_revise.append([
                    {"role":"user",
                     "content": REVISE_PROMPT_ICL.format(prompt[i][-1]["content"], j) 
                     }])
    else:
        raise

    if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
        logging.info("[Revise Prompt]\n%s" % (best_text_revise))

    best_text_revise = apply_template(args.pretrained_model, best_text_revise, tokenizer, add_generation_prompt=True) 
    best_text_revise = tokenizer(best_text_revise, return_tensors="pt", padding='longest')
    with torch.no_grad():
        generate_ids = model.generate(**best_text_revise.to(device), do_sample=False, max_new_tokens=2048, stopping_criteria=stop_criteria) # max_length=args.length + model_inputs.shape[-1] #stopping_criteria=stop_criteria
    pure_generate_ids = generate_ids[:, best_text_revise['input_ids'].shape[1]:]
    best_text_revise = tokenizer.batch_decode(pure_generate_ids, skip_special_tokens=True)
    

    final=[]
    for i in range(num_sample):
        final.extend([prompt[i] +  [{"role":'assistant','content':j}] for j in  best_text[i*args.batch_size: (i+1) * args.batch_size]])
    for i in range(num_sample):
        final.extend([prompt[i] +  [{"role":'assistant','content':j}] for j in  best_text_revise[i*args.batch_size: (i+1) * args.batch_size]])

    final_reward_score = compute_reward_score(args, reward_tokenizer, reward_model, final, device=device)
    # final = apply_template(args.reward_model, final, reward_tokenizer, add_generation_prompt=False)
    # with torch.no_grad():
    #     kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
    #     final = reward_tokenizer(final, **kwargs).to(device)
    #     final_reward_score = reward_model(**final).logits

    hard_reward_score=final_reward_score[:int(final_reward_score.shape[0]/2)].squeeze(-1)
    hard_revised_reward_score=final_reward_score[int(final_reward_score.shape[0]/2):].squeeze(-1)

    if (not args.accelerator) or (args.accelerator and accelerator.is_main_process):
        logging.info("[Optimized] Reward: %s\n%s" % (hard_reward_score, best_text))
        logging.info("[Revised Optimized] Reward: %s\n%s" % (hard_revised_reward_score, best_text_revise))
        logging.info("[Time]: %s s\n\n" % (elapsed_time))
    
    ret=[]
    for i in range(num_sample):
        ret.append({"text": best_text[i*args.batch_size: (i+1) * args.batch_size],
                    "soft_reward_score": best_reward_score[i*args.batch_size: (i+1) * args.batch_size].tolist(),
                    "hard_reward_score": hard_reward_score[i*args.batch_size: (i+1) * args.batch_size].tolist(),
                    "revised_text": best_text_revise[i*args.batch_size: (i+1) * args.batch_size],
                    "revised_reward_score": hard_revised_reward_score[i*args.batch_size: (i+1) * args.batch_size].tolist(),
                    "original_output": original_hard_output[i], 
                    "original_output_reward_score": original_hard_output_reward_score[i], 
                    "chosen_given": chosen[i][-1]["content"] if (len(chosen) > 0 and (None not in chosen))  else None, 
                    "chosen_given_reward_score": chosen_given_reward_score[i] if (len(chosen) > 0 and (None not in chosen)) else None,
                    "rejected_given": rejected[i][-1]["content"] if ((len(rejected)> 0) and (None not in rejected)) else None, 
                    "rejected_given_reward_score": rejected_given_reward_score[i] if ((len(rejected)> 0) and (None not in rejected)) else None,
                    "best_iter": best_iter[i*args.batch_size: (i+1) * args.batch_size].tolist()
                    })        
    return ret, elapsed_time