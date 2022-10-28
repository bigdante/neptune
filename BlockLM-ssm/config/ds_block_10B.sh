#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_block_10B.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 1.0 \
       --gap-sentence-prob 0.0 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.0 \
       --gpt-infill-prob 0.0 \
       --block-mask-prob 0.1 \
       --experiment-name blocklm-10b-ssm-short \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 48 \
       --hidden-size 4096 \
       --num-attention-heads 64 \
       --seq-length 513 \
       --max-sequence-length 1025 \
       --save /dataset/fd5061f6/english_data/checkpoints \
       --load /dataset/fd5061f6/english_data/checkpoints/blocklm-10b-512 \
       --new-save-directory \
       --old-checkpoint \
       --no-load-optim \
       --log-interval 50 \
       --eval-interval 200 \
       --save-interval 200 \
       --train-iters 250000 \
       --train-data wikipedia_ssm \
       --resume-dataloader \
       --filter-english \
       --tokenizer-type GPT2BPETokenizer \
       --tokenizer-model-type gpt2 \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-ratio 0.1 \
       --lr-decay-iters 175000 \
       --warmup 0.04 \
       --checkpoint-activations \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"
