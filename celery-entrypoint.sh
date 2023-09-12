#!/bin/bash

set -ex

#export CUDA_VISIBLE_DEVICES="2,3,4,5"
#export NCCL_DEBUG=INFO # TRACE
##export NCCL_PROTO=SIMPLE
##
##rank=$1
##
##master_id=5
##node_id=$((rank + master_id))
##local_hostname=$(hostname)
#
#
##export LOGLEVEL="DEBUG"
#export LOGLEVEL="INFO"
#
#/mnt/ess_storage/DN_1/storage/qa-system-research/zakharova/py_envs/transformers-py38-seq2seq/bin/python -m torch.distributed.launch \
#    --nproc_per_node 4 \
#    serving/celery_qa_serving_task.py

/py_envs/transformers-py38-seq2seq/bin/python -m torch.distributed.launch \
    --nproc_per_node ${NUM_OF_DISTR_WORKERS} \
    serving/celery_qa_serving_task.py


#/mnt/ess_storage/DN_1/storage/qa-system-research/zakharova/py_envs/transformers-py38-seq2seq/bin/python -m celery purge
#torchrun \
#    --nproc_per_node 1 \
#    --nnodes ${WORLD_SIZE} \
#    --node_rank ${rank} \
#    --master_addr "node${master_id}.bdcl" \
#    --master_port=10234 \
#    --local_addr="node${node_id}.bdcl" \
#    --rdzv_id=llama_job_1 \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint="node${master_id}.bdcl:29400" \
#    /src/serving/celery_qa_serving_task.py


#celery -A serving.celery_qa_serving_task worker --loglevel=INFO