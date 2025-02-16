#!/bin/bash
DEFAULT_PORT=29524
DEFAULT_GPUS="0,1,2,3,4,5,6,7"
PORT=${1:-$DEFAULT_PORT}
GPUS=${2:-$DEFAULT_GPUS}

export CUDA_VISIBLE_DEVICES=$GPUS
export WANDB_MODE=offline

flu_weights=(0.1)
lrs=(0.01)
reward_temps=(0.1)
init_temps=(10)
init_modes=("original")
topks=(10)

for flu_weight in "${flu_weights[@]}"; do
  for lr in "${lrs[@]}"; do
    for reward_temp in "${reward_temps[@]}"; do
      for init_temp in "${init_temps[@]}"; do
        for init_mode in "${init_modes[@]}"; do
          for topk in "${topks[@]}"; do
            echo "Running with flu-weight=$flu_weight, lr=$lr, reward-temp=$reward_temp, init-temp=$init_temp, init-mode=$init_mode, topk=$topk"

            accelerate launch --main_process_port $PORT  main.py \
              --description none \
              --reward-model RM-llama-3.2-3b \
              --pretrained-model llama-3.2-3b-base \
              --dataset advbench \
              --length 50 \
              --num-iters 200 \
              --max-length 2048 \
              --min-length 50 \
              --reward-weight 1 \
              --flu-weight $flu_weight \
              --end-weight 0 \
              --lr $lr \
              --noise-iters 1 \
              --win-anneal-iters 1000 \
              --topk $topk \
              --reward-topk 0 \
              --div RKL \
              --detach \
              --init-temp $init_temp \
              --reward-temp $reward_temp \
              --straight-through \
              --large-noise-iters 50,100,150,200 \
              --large-gs-std 0.1,0.05,0.01,0.001 \
              --batch-size 4 \
              --outter-batch-size 8 \
			        --init-mode $init_mode \
			        --revise-mode A \
              --print-every 10 \
              --seed 24 \
              --wandb \
              --verbose \
              --patience 100 \
              --start 0 \
              --end 100000 \
              --fp16 \
              --verify \
              --accelerator \
              --ddp
          done
        done
      done
    done
  done
done

