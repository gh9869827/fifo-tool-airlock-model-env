# Airlock Model Environment

A more secure, fully isolated environment for running Hugging Face models that require `trust_remote_code=True`. This architecture ensures maximum containment using a no-network Docker container, communicating only through stdin/stdout with a local FastAPI bridge.

> ⚠️ **Note**: This setup is not intended to be scalable but provide a more secure way to run models requiring `trust_remote_code=True` by restricting network access inside the container where the model runs. The SDK communicates with the container by using `docker exec` and by using stdin and stdout.

This projects currently focuses on Phi4 models: they are lightweight, can be run on consumer GPU, capable for agentic deployments and currently requires `trust_remote_code=True`.
---

## 🧩 Architecture Overview

```
      ┌────────────┐
      │   User /   │  ← External tools / apps
      │   Client   │  ← e.g., Python SDK, CLI tool, chatbot
      └────┬───────┘
           │
      ┌────▼──────┐
      │  Bridge   │  ← FastAPI server (127.0.0.1) ⇄ docker exec client
      └────┬──────┘
           │
      ┌────▼────────────────────────────┐
      │  Docker Container               │  ← Isolated, no network
      │  ┌────────────────────────────┐ │
      │  │  Airlocked client          │ │  ← Receives input from docker exec via stdin
      │  └────────────────────────────┘ │     ⇅ Communicates with Airlocked Server via http requests
      │                                 │     ⇡ Returns response to Bridge via stdout
      │  ┌────────────────────────────┐ │
      │  │  Airlocked Model Server    │ │  ← FastAPI server (127.0.0.1):
      │  └────────────────────────────┘ │     Persistent
      └─────────────────────────────────┘     Loads HF models, processes input
```

---

## 📁 Directory Structure

```
airlock_model_env/
├── server/            # Container-side FastAPI server that provides the Airlocked model server logic (runs inside Docker)
│   ├── fastapi_server.py
│   ├── llm_model.py
│   ├── llm_model_phi_4.py
│   ├── llm_model_phi_4_mini_instruct.py
│   ├── llm_model_phi_4_multimodal_instruct.py
│   └── __init__.py
├── client/            # CLI entrypoint executed via `docker exec`, sends requests via stdin/stdout
│   ├── run.py
│   └── __init__.py
├── bridge/            # Host-side FastAPI bridge server, proxies requests to Docker-contained model
│   ├── fastapi_server.py
│   └── __init__.py
├── sdk/               # Client SDK to talk to the FastAPI bridge
│   ├── client_sdk.py
│   └── __init__.py
├── common/            # Shared request/response schemas and data models
│   ├── models.py
│   └── __init__.py
└── README.md          # You're here!
```

---

## ⚠️ Prerequisites

The container is assumed to have a working PyTorch environment with CUDA and cuDNN properly installed to run Hugging Face models.

## 📦 Python Dependencies

Install essential packages for working with large language models and serving them using application servers:

```bash
python3 -m pip install --upgrade pip setuptools wheel

python3 -m pip install uvicorn fastapi

python3 -m pip install transformers accelerate bitsandbytes peft backoff flash_attn pydantic
```

### 🔍 Package Summary

| Package        | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `transformers` | State-of-the-art pre-trained models for NLP, vision, and beyond.            |
| `accelerate`   | Simplifies training and inference on CPUs, GPUs, or multi-GPU setups.       |
| `bitsandbytes` | Lightweight quantization and 8-bit optimizations for large models.          |
| `peft`         | Tools for parameter-efficient fine-tuning (LoRA, AdaLoRA, etc.) of LLMs.    |
| `trl`          | Fine-tuning trainer utilities for supervised and reinforcement learning. *(if fine-tuning)* |
| `datasets`     | Hugging Face's standard for dataset handling and preprocessing. *(if fine-tuning)* |
| `backoff`      | Retry logic with exponential backoff — helpful for flaky APIs.              |
| `flash-attn`   | Fast CUDA-optimized attention for large transformers.                       |
| `pydantic`     | Type-validated models for structured data — used heavily in FastAPI.        |
| `fastapi`      | High-performance web framework for building APIs with async Python.         |
| `uvicorn`      | Lightning-fast ASGI server for serving FastAPI apps in production.          |

---

## 🔐 Authentication

Login to Hugging Face:

```bash
huggingface-cli login
```

---

## 📥 Download Models (Phi-4 only)

```bash
huggingface-cli download microsoft/Phi-4-mini-instruct
huggingface-cli download microsoft/Phi-4-multimodal-instruct
```

- Disconnect from the network after downloading the models

```bash
docker network disconnect bridge dev-phi
```

---

## 🚀 Quick Start

### 1. Start the Docker container

```bash
docker run --rm \
  --network=none \
  --name airlock \
  airlock-image
```

### 2. Run the FastAPI bridge (on host)
From project root:

```bash
uvicorn bridge.fastapi_server:app --host 127.0.0.1 --port 8000
```

### 3. Use the client SDK (or your own tool)

```python
from sdk.client_sdk import call_airlock_model_server
response = call_airlock_model_server("What is the capital of France?")
print(response)
```

---

## 🔒 Security Notes
- The model runs with `trust_remote_code=True`, but is fully isolated
- No data is persisted inside the container
- Communication is limited to `stdin` and `stdout`
- Bridge can enforce access control, rate limiting, etc.

---

## 📦 Goals
- Minimal trusted surface area
- Clean interface between components
- Easy to extend or audit

---

## ✅ License
MIT
