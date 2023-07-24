#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES="0"

rank=$1
node_id=$((rank + 1))

export LOGLEVEL="DEBUG"

torchrun \
    --nproc_per_node 1 \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${rank} \
    --master_addr node1.bdcl \
    --master_port=10234 \
    --local_addr="node${node_id}.bdcl" \
    --rdzv_id=llama_job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=node1.bdcl:29400 \
    example_text_completion.py \
    --ckpt_dir ${CKPT_DIR_PATH} \
    --tokenizer_path /llama/tokenizer.model \
    --max_seq_len 128 --max_batch_size 1
