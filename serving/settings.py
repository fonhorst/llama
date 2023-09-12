from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    ckpt_dir: str = '/mnt/ess_storage/DN_1/storage/home/khodorchenko/LM/llama/llama-2-13b-chat-4shards/'
    tokenizer_path: str = '/mnt/ess_storage/DN_1/storage/home/khodorchenko/LM/llama/tokenizer.model'
    redis_host: str = "node9.bdcl"
    redis_port: int = 6379
    redis_streams_db: int = 1
    redis_streams_answer_stream: str = 'llama_test_stream2'
    redis_celery_db: int = 0
    redis_celery_backend: int = 3
    temperature: float = 0.3
    top_p: float = 0.9
    max_seq_len: int = 2048
    max_gen_len: int = 512
    max_batch_size: int = 1

    model_config = SettingsConfigDict(
        env_file=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '.env'
        ),
        env_file_encoding='utf-8'
    )


settings = Settings()
