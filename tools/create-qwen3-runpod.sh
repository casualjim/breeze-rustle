#!/usr/bin/env bash

runpodctl create pod \
  --gpuType 'NVIDIA GeForce RTX 3090' \
  --imageName vllm/vllm-openai:latest \
  --volumeSize 30 \
  --containerDiskSize 10 \
  --name qwen3-embed \
  --secureCloud \
  --args '--host 0.0.0.0 --port 8000 --model Qwen/Qwen3-Embedding-0.6B --enforce-eager --trust-remote-code --max-model-len 32768 --max-num-seqs 200 --max-num-batched-tokens 192000 --gpu-memory-utilization 0.95 --disable-log-stats --api-key sk-moFgaeiq8K76FXkCNx2ydwtnxWutYotT' \
  --ports '8000/http' \
  --ports '22/tcp' \
  --volumePath /workspace \
  --env "HF_HOME=/workspace/hf_home"
