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
            --load-pretrained /raid/xll/checkpoints/blocklm-10b-ssm/113400"
