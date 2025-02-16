from transformers import LlamaPreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
import torch
from typing import Optional, List


def load_reward_model_and_tokenizer(model_path, device=None, args=None):
    if args.reward_model == "RM-llama-2-7b-oasst1":
        config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            device_map=device, 
            torch_dtype=torch.float16 if args.fp16 else torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.flash_atten_2 else None
        ).eval()
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_auth_token=True)
    elif args.reward_model == "RM-ultrarm-13b":
        model = LlamaRewardModel.from_pretrained(
            model_path,
            device_map=device,
            attn_implementation="flash_attention_2" if args.flash_atten_2 else None
        ).eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16 if args.fp16 else torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.flash_atten_2 else None
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # print('model.config.eos_token_id',model.config.eos_token_id)
    # for i in model.config.eos_token_id:
    #     print(tokenizer.decode(i))
    # exit(0)
    model.config.pad_token_id = model.config.eos_token_id
    if type(model.config.eos_token_id) is list:
        model.config.pad_token_id = model.config.eos_token_id[0]
    # else:
    #     model.config.pad_token_id = None
    
    return model, tokenizer


def load_model_and_tokenizer(model_path, device=None, args=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16 if args.fp16 else torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.flash_atten_2 else None
    ).eval()
    # tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    if type(model.config.eos_token_id) is list:
        model.config.pad_token_id = model.config.eos_token_id[0]
    # if ('llama-3' not in model_path.lower()) and ('llama3' not in model_path.lower()):
    #     model.config.pad_token_id = model.config.eos_token_id
    # else:
    #     model.config.pad_token_id = None

    return model, tokenizer


class Reward:
    def __init__(self):
        self.logits = None
        self.past_key_values = None

class LlamaRewardModel(LlamaPreTrainedModel):
    config_class = LlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward( # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        ret = Reward()
        transformer_outputs = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,                               
                            )
        # print('attention_mask',attention_mask)
        # print('attention_mask',attention_mask.shape)
        # print('transformer_outputs',len(transformer_outputs))
        hidden_states = transformer_outputs[0]
        # print('hidden_states',hidden_states.shape)
        rewards = self.regression_head(hidden_states).squeeze(-1)
        # print('rewards',rewards.shape)
        
        if attention_mask is None:
            ends=torch.Tensor([[rewards.shape[-1]-1]]).type(torch.int64)
        else:
            ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)
        ret.logits = torch.gather(rewards, 1, ends.to(rewards.device))
        ret.past_key_values = transformer_outputs.past_key_values
        
        return ret