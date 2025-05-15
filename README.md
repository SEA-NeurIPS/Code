# Inference-time Alignment in Continuous Space

This repository provides the implementation code for our submission to the NeurIPS 2025: Inference-time Alignment in Continuous Space. In this paper, we propose SEA, a simple inference-time alignment algorithm that reformulates alignment as an iterative optimization procedure on an energy function over logits in the continuous space defined by the optimal RLHF policy for deep and effective alignment. Despite its simplicity, SEA enjoys promising performance on extensive benchmarks such as AdvBench and TruthfulQA, consistently and significantly outperforming state-of-the-art baselines across various base models. 


## Environment
Create a Python virtual environment using e.g. Conda:

```shell
conda create -n sea python=3.10 && conda activate sea
```

First, install PyTorch `2.1.2` from the [PyTorch Installation Page](https://pytorch.org/get-started/locally/).

Then, install the following packages:
```shell
pip install -r requirements.txt
```

## Inference
See scripts, for example:
```
bash scripts/adv-llama3.2-1b-base.sh # Default Accelerate Port & GPU id
bash scripts/adv-llama3.2-1b-base.sh 29520 # Default Accelerate Port & Default GPU id
bash scripts/adv-llama3.2-1b-base.sh 29520 "2,4" # Specified Accelerate Port & GPU id
```
## Evaluation
See scripts, for example:
```
bash scripts/eval.sh
```

## Outputs
See outputs
