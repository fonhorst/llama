## Llama-2 13b chat service

This branch purposed to run on a DGX. 
To launch a container with 2 celery workers which process websocket queries:
1. Configure `docker-compose.yml` file:
```yaml
  # Launching script configuration
  CUDA_VISIBLE_DEVICES: "4,5" # Two GPUs for 13b 
  NUM_OF_DISTR_WORKERS: 2 # Number of available GPUs
  LOGLEVEL: "INFO"
  CKPT_DIR: /llama/llama-2-13b-chat/ # Path to the model
  TOKENIZER_PATH: /llama/tokenizer.model # Path to the tokenizer
  # Generation parameters
  TEMPERATURE: 0.3 # Generation param: temperature
  TOP_P: 0.9 # Generation param: top_p
  MAX_SEQ_LEN: 2048 # Generation param: maximum sequence length
  MAX_GEN_LEN: 1024 # Generation param: maximum new generated tokens length 
  MAX_BATCH_SIZE: 1 # Currently only one user can query the model, leave it equal to 1
  # API compatible parameters (Should be set according to the params in llama2-api containers
  REDIS_HOST: "node9.bdcl" # Host of the Redis db, holding streams, celery backend, celery broker
  REDIS_PORT: 6380 # Port of the Redis db, holding streams, celery backend, celery broker
  REDIS_STREAMS_DB: 1 # Redis streams database number
  REDIS_STREAMS_ANSWER_STREAM: answer_stream # Redis streams stream name (should be the same as the stream, read by API)
  REDIS_CELERY_DB: 0 # Redis celery broker database
  REDIS_CELERY_BACKEND: 3 # Redis celery backend database
```
2. Syncronize code with the dgx:
```bash
HOST_NAME=your_username  SYNC_HOST=host_to_which_sync ./rsync-repo.sh upload
```
3. On the DGX run:
```bash
docker-compose build
docker-compose up -d
```
4. Check that everything is ok and Celery workers started:
```bash
docker logs --follow llama_llm_1
```
The model must be loaded to GPU, two celery workers are available and `run_completion` task is shown among tasks:
```bash
[tasks]
  . run_completion
```