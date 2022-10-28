device=`expr $1 - 21534`
tmux new-session -s $device "export CUDA_VISIBLE_DEVICES=${device} && cd /raid/liuxiao/BlockLM-ssm && bash scripts/generate_ent_batch.sh config_tasks/model_blocklm_10B_ent.sh ${1}; exec bash -i"
