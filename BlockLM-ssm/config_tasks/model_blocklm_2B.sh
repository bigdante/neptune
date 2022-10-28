MODEL_TYPE="blocklm-2B"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 36 \
            --hidden-size 2048 \
            --num-attention-heads 32 \
            --max-sequence-length 1025 \
	    --tokenizer-type GPT2BPETokenizer \
            --tokenizer-model-type gpt2 \
            --old-checkpoint \
            --load-pretrained /raid/liuxiao/checkpoints/blocklm-2b-512/170000"
