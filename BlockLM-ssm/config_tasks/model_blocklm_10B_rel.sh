MODEL_TYPE="blocklm-10B"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 48 \
            --hidden-size 4096 \
            --num-attention-heads 64 \
            --max-sequence-length 1025 \
	          --tokenizer-type GPT2BPETokenizer \
            --tokenizer-model-type gpt2 \
            --old-checkpoint \
            --load-pretrained /raid/liuxiao/checkpoints/blocklm-10b-512"
