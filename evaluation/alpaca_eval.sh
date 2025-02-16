model_outputs=$1
reference_outputs=$2
output_path=$3
max_instances=$4

annotators_config="weighted_alpaca_eval_gpt4_turbo"
is_overwrite_leaderboard="True"

alpaca_eval --model_outputs $model_outputs \
            --reference_outputs $reference_outputs \
            --output_path $output_path \
            --annotators_config $annotators_config \
            --is_overwrite_leaderboard $is_overwrite_leaderboard \
            --max_instances $max_instances 
