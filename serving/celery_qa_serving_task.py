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
from celery.signals import worker_process_shutdown, worker_process_init, task_revoked


@dataclass
class GenerationArguments:
    temperature: float
    top_p: float
    max_generation_length: int
    max_sequence_length: int
    batch_size: int


rs: Redis = redis.Redis(host=settings.redis_host, port=settings.redis_port, db=settings.redis_streams_db)  # TODO

app = Celery(
    'celery_app',
    broker=f'redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_celery_db}',
    result_backend=f'redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_celery_backend}'
)
app.conf.result_backend = f'redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_celery_backend}'
app.conf.CELERY_ENABLE_UTC = True

generator = None
generation_args = None


@app.task(name='run_completion', soft_time_limit=60)
def run_completion(
        input_text: str,
        stream_name: str = settings.redis_streams_answer_stream,
):
    try:
        print(f'Running completion {str(os.environ["LOCAL_RANK"])}')
        results = generator.text_completion(
            [input_text],
            max_gen_len=generation_args.max_generation_length,
            temperature=generation_args.temperature,
            top_p=generation_args.top_p,
            put_results_to_redis_streams=rs,
            stream_name=stream_name
        )
        if int(os.environ["LOCAL_RANK"]) == 0:
            print(results)

    except Exception as exc:
        print(exc)

    return True


@task_revoked.connect(sender=run_completion)
def run_completion_on_revoke(sender=None, request=None, terminated=None, signum=None, expired=None, **kwargs):
    if int(os.environ["LOCAL_RANK"]) == 0:
        rs.xadd(
            request.kwargs.get('stream_name'),
            {'text': '', 'is_eos': -1}
        )


@worker_process_init.connect
def init_worker(**kwargs):
    app.control.purge()
    # TODO flush redis? unacked unacked_index

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


@worker_process_shutdown.connect
def shutdown_worker(**kwargs):
    app.control.purge()
    # TODO flush redis?
    # Cleanup GPU and other resources here
    # This code runs once for each worker process when it shuts down.


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    local_rank = str(os.environ["LOCAL_RANK"])
    args = ['worker', '--loglevel=INFO', '-n', local_rank, '-P', 'solo']
    app.worker_main(argv=args)
