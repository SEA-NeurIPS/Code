export OPENAI_API_BASE="https://hb.rcouyi.com/v1"
export OPENAI_API_KEY='sk-FSOVn4hAIghgnJ0PA34c978a6f55446c918653D26fDd5fC4'

# python gen_model_answer.py --model-path  $name  --model-id $modelid
modelid="llama-3.2-1b-base_iea_NewPrompt_RM-llama-3.2-3b_bs1-original_RKL_len0-max512-min50_e200_adam-lr0.005-200-1_weight-f0.1-r1.0-e0.0_topk1000-r0_temp-i10.0-r0.05_reQA"
python gen_judgment.py --model-list $modelid --parallel 10 --first-n 10
