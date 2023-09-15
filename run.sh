#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES="0"
export NCCL_DEBUG=TRACE
export NCCL_PROTO=SIMPLE
##export NCCL_DEBUG_SUBSYS=COLL
#export NCCL_SOCKET_IFNAME="=vlan501"

#echo "NCCL_PROTO ${NCCL_PROTO}"
#echo "NCCL_ALGO ${NCCL_ALGO}"

rank=$1

master_id=5
#master_id=9
node_id=$((rank + master_id))
local_hostname=$(hostname)


export LOGLEVEL="DEBUG"

torchrun \
    --nproc_per_node 1 \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${rank} \
    --master_addr "node${master_id}.bdcl" \
    --master_port=10234 \
    --local_addr="node${node_id}.bdcl" \
    --rdzv_id=llama_job_1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="node${master_id}.bdcl:29400" \
    example_text_completion.py \
    --ckpt_dir ${CKPT_DIR_PATH} \
    --tokenizer_path /llama/tokenizer.model \
    --max_seq_len 4096 --max_gen_len=512 --max_batch_size 2 --temperature 0.3 \
    --prompts_directory /mnt/ess_storage/DN_1/storage/qa-system-research/zakharova/test_folder_prompts_subscribtion1 \
    --prediction_files_dir /mnt/ess_storage/DN_1/storage/qa-system-research/zakharova/test_folder_prompts_outputs1 \
    --mask_tensor_path /llama/tensor-tokens-sber-domain_v14.pt


#torchrun \
#    --nproc_per_node 1 \
#    --nnodes ${WORLD_SIZE} \
#    --node_rank ${rank} \
#    --master_addr "node${master_id}.bdcl" \
#    --master_port=10234 \
#    --local_addr=${local_hostname} \
#    --rdzv_id=llama_job \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint="node${master_id}.bdcl:29400" \
#    example_text_completion.py \
#    --ckpt_dir ${CKPT_DIR_PATH} \
#    --tokenizer_path /llama/tokenizer.model \
#    --max_seq_len 2048 --max_gen_len=512 --max_batch_size 2 --temperature 0.3 \
#    --prompts_directory /mnt/ess_storage/DN_1/storage/qa-system-research/zakharova/70b_prompts_subscribtion \
#    --prediction_files_dir /mnt/ess_storage/DN_1/storage/qa-system-research/zakharova/70b_prompts_outputs

#torchrun \
#    --nproc_per_node 1 \
#    --nnodes ${WORLD_SIZE} \
#    --node_rank ${rank} \
#    --master_addr "node${master_id}.bdcl" \
#    --master_port=10234 \
#    --local_addr=${local_hostname} \
#    --rdzv_id=llama_job \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint="node${master_id}.bdcl:29400" \
#    example_text_completion.py \
#    --ckpt_dir ${CKPT_DIR_PATH} \
#    --tokenizer_path /llama/tokenizer.model \
#    --max_seq_len 2048 --max_gen_len=512 --max_batch_size 2 --temperature 0.3 \
#    --prompts_directory /mnt/ess_storage/DN_1/storage/qa-system-research/zakharova/13b_prompts_subscribtion \
#    --prediction_files_dir /mnt/ess_storage/DN_1/storage/qa-system-research/zakharova/13b_prompts_outputs