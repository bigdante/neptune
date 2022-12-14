#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_block_220M.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.5 \
       --gap-sentence-prob 0.3 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --experiment-name blocklm-220M-ssm \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 14 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --seq-length 512 \
       --max-sequence-length 513 \
       --save /dataset/fd5061f6/english_data/checkpoints \
       --train-iters 200000 \
       --resume-dataloader \
       --train-data wikipedia_ssm \
       --tokenizer-type GPT2BPETokenizer \
       --tokenizer-model-type gpt2 \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-iters 160000 \
       --lr-decay-ratio 0.05 \
       --warmup .05 \
       --checkpoint-activations \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"
