#!/bin/bash
export OPENAI_API_KEY=''

OUTPUT_DIR="outputs"
RM_MODEL='RM-llama-3.2-3b'
RESULTS_DIR="${OUTPUT_DIR}/results"
mkdir -p "$RESULTS_DIR"

BASE_MODELS=('llama-3.2-1b-base' 'llama-3.2-1b-instruct' 'llama-3.2-3b-base' 'llama-3-8b-base') 
DATASETS=('advbench')

for BASE_MODEL in "${BASE_MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    if [[ "$DATASET" == "advbench" ]]; then
      NUM_SAMPLE=520
    elif [[ "$DATASET" == "tqa" ]]; then
      NUM_SAMPLE=100
    else
      NUM_SAMPLE=200
    fi
    echo "Running evaluation for Base Model: $BASE_MODEL, Dataset: $DATASET, Num-Sample: $NUM_SAMPLE"
    CUDA_VISIBLE_DEVICES=5,6 python -u eval.py \
        --path ./${OUTPUT_DIR}/${DATASET}/${BASE_MODEL}/${RM_MODEL}/ \
        --num-sample $NUM_SAMPLE >> ./${RESULTS_DIR}/${DATASET}_${BASE_MODEL}_${NUM_SAMPLE}.txt
  done
done
