
source $1

MPSIZE=1
MAXSEQLEN=384
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=1
TOPP=0

#python fewrel_ent_serving_desc_multi.py


python -m torch.distributed.launch --nproc_per_node=$MPSIZE --master_port $MASTER_PORT fewrel_ent_serving_desc_multi.py\
       --mode inference \
       --length-penalty 0.7 \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP \
       --out-seq-length $MAXSEQLEN \
       --model-parallel-size $MPSIZE \
       $MODEL_ARGS \
       --fp16 \
       --batch-size 2 \
       --inference-strategy-constrained \
       --serving-port $2 \

