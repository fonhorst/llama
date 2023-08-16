# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import datetime
from dataclasses import dataclass
import json
import logging
import os
from typing import Optional
import time

import fire
import numpy as np
from tqdm.auto import tqdm
# import torch
import torch.distributed as dist
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler

from llama_dgx import Llama

@dataclass
class FilePrompts:
    path_to_file: str
    prompts: dict


@dataclass
class GenerationArguments:
    temperature: float
    top_p: float
    max_generation_length: int
    max_sequence_length: int
    batch_size: int


def run_completion(model: Llama, prompts: FilePrompts, generation_args: GenerationArguments):
    # local_rank = int(os.environ["GROUP_RANK"])
    start = datetime.datetime.now()
    logging.warning(f'Prompt file: {prompts.path_to_file}')
    num_batches = int(np.ceil(len(prompts.prompts) / generation_args.batch_size))
    predictions = {}
    for batch in tqdm(range(num_batches)):
        prompt_keys = range(batch * generation_args.batch_size, (batch + 1) * generation_args.batch_size)
        input_prompts = []
        input_keys = []
        for key in prompt_keys:
            input = prompts.prompts.get(str(key))
            if input is not None:
                input_prompts.append(input)
                input_keys.append(key)
        results = model.text_completion(
            input_prompts,
            max_gen_len=generation_args.max_generation_length,
            temperature=generation_args.temperature,
            top_p=generation_args.top_p,
        )
        # logging.warning(f"Batch done")
        for prompt_key, prediction_res in zip(input_keys, results):
            predictions[prompt_key] = prediction_res
    end = datetime.datetime.now()
    logging.warning(f"Execution time: {end - start} secs")
    return predictions


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 128,
        max_gen_len: int = 64,
        max_batch_size: int = 4,
        prompts_directory: Optional[str] = None,
        prediction_files_dir: Optional[str] = None,
):
    local_rank = int(os.environ["LOCAL_RANK"])
    logging.warning(f'Rank: {local_rank}')
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    generation_args = GenerationArguments(
        temperature=temperature,
        top_p=top_p,
        max_generation_length=max_gen_len,
        max_sequence_length=max_seq_len,
        batch_size=max_batch_size,
    )
    checked_files = set(os.listdir(prediction_files_dir))
    while True:
        try:
            files_to_check = list(set(os.listdir(prompts_directory)) - checked_files)
            prompt_files = []
            for file in sorted(files_to_check):
                logging.warning(f"Loading: {file}")
                try:
                    with open(os.path.join(prompts_directory, file), 'r') as f:
                        retrieved_prompts = json.load(f)
                    prompt_files.append(FilePrompts(path_to_file=file, prompts=retrieved_prompts))
                except json.JSONDecoder as exc:
                    logging.warning(f"{file} could not be loaded")
                    continue
                finally:
                    checked_files.add(file)
            # else:
            #     prompt_files = [None] * len(files_to_check)
            # dist.broadcast_object_list(prompt_files, src=0)
            if not prompt_files:
                logging.warning("Waiting for new files...")
                time.sleep(60)
                continue
            else:
                logging.warning(f"Running completion on {len(prompt_files)} files.")
            for prompt in prompt_files:
                predictions = run_completion(
                    generator,
                    prompt,
                    generation_args,
                )
                if local_rank == 0:
                    with open(os.path.join(prediction_files_dir, os.path.basename(prompt.path_to_file)), 'w') as f:
                        json.dump(predictions, f)

        except Exception as exc:
            raise ValueError("Something went wrong") from exc

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    fire.Fire(main)
