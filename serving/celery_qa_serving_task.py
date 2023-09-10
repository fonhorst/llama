# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
import logging
import os
import redis
# from llama.settings import settings
from llama_dgx import Llama
from settings import settings
from redis import Redis
from celery import Celery
import torch
# from celery.signals import worker_process_init

@dataclass
class GenerationArguments:
    temperature: float
    top_p: float
    max_generation_length: int
    max_sequence_length: int
    batch_size: int

rs: Redis = redis.Redis(host=settings.redis_host, port=settings.redis_port, db=settings.redis_streams_db) # TODO

app = Celery(
    'celery_app',
     broker=f'redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_celery_db}',
     result_backend=f'redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_celery_db}'
 )



generator = None
generation_args = None
# local_rank = int(os.environ["GROUP_RANK"])
# logging.warning(f'Rank: {local_rank}')

# print(os.listdir(settings.tokenizer_path))
# @worker_process_init.connect()
def init_models():
    global generator
    global generation_args
    generator = Llama.build(
        ckpt_dir=settings.ckpt_dir,
        tokenizer_path=settings.tokenizer_path,
        max_seq_len=settings.max_seq_len,
        max_batch_size=settings.max_batch_size,
    )
    generation_args = GenerationArguments(
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_generation_length=settings.max_gen_len,
        max_sequence_length=settings.max_seq_len,
        batch_size=settings.max_batch_size,
    )


@app.task(name='run_completion')
def run_completion(
        input_text: str,
):
    print(f'Running completion {str(os.environ["LOCAL_RANK"])}!')
    results = generator.text_completion(
        [input_text],
        max_gen_len=generation_args.max_generation_length,
        temperature=generation_args.temperature,
        top_p=generation_args.top_p,
        put_results_to_redis_streams=rs,
    )

    return results

# app.conf.task_routes = {
#     'run_completion': {'queue': 'llama'},
# }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # torch.multiprocessing.set_start_method('spawn')
    init_models()
    local_rank = str(os.environ["LOCAL_RANK"])
    args = ['worker', '--loglevel=INFO', '-n', local_rank, '-P', 'solo']
    app.worker_main(argv=args)
    # worker = app.Worker(include=['serving.celery_qa_serving_task', 'serving.llama_dgx'])
    # worker.start()





