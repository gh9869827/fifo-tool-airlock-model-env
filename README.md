# Airlock Model Environment

A more secure, fully isolated environment for running Hugging Face models that require `trust_remote_code=True`. This architecture ensures maximum containment using a no-network Docker container, communicating only through stdin/stdout with a local FastAPI bridge.

> ⚠️ **Note**: This setup is not intended to be scalable but provide a more secure way to run models requiring `trust_remote_code=True` by restricting network access inside the container where the model runs.
The SDK communicates with the container via `docker exec`, using `stdin` and `stdout` for transport

This projects currently focuses on Phi4 models: they are lightweight, can be run on consumer GPU, capable for agentic deployments and currently requires `trust_remote_code=True`.
For each base model, additional LoRA adapters can be dynamically loaded, allowing multiple variants to share a single model backbone with efficient memory usage.
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

Here's an overview of the project layout, showing the separation between host and container components:

```
fifo_tool_airlock_model_env/
├── bridge/                                      # Host-side FastAPI bridge server
│   ├── __init__.py
│   └── fastapi_server.py
├── client/                                      # CLI entrypoint (invoked via docker exec)
│   ├── __init__.py
│   └── run.py
├── common/                                      # Shared schemas and models
│   ├── __init__.py
│   └── models.py
├── examples/                                    # Example scripts
│   ├── call_model.py
│   └── call_multimodal_model.py
├── sdk/                                         # Client SDK for interacting with bridge
│   ├── __init__.py
│   └── client_sdk.py
├── server/                                      # Airlocked model server logic (container-side)
│   ├── __init__.py                              #   Package initializer
│   ├── fastapi_server.py                        #   Launches the container-side FastAPI app
│   ├── logging_config.py                        #   Logging configuration for Uvicorn
│   ├── llm_model_loader.py                      #   Entrypoint to load models from config
│   ├── llm_model.py                             #   Abstract base class for model implementations
│   ├── llm_model_phi_4_base.py                  #   Shared core logic for Phi-4 family
│   ├── llm_model_phi_4_base_with_adapters.py    #   Adds LoRA adapter handling to base loader
│   ├── llm_model_phi_4_mini_instruct.py         #   Loader for Phi-4 Mini Instruct with adapter support
│   └── llm_model_phi_4_multimodal_instruct.py   #   Loader for Phi-4 Multimodal with adapter support
├── model_config.example.json                    # Config template
├── pyproject.toml                               # Build and dependency metadata
├── LICENSE                                      # MIT license
└── README.md                                    # You're here!
```

---
## ⚠️ Prerequisites

The container is assumed to have **PyTorch 2.6.0+cu126**, with matching **CUDA 12.6** and **cuDNN**, installed to support Hugging Face models with GPU acceleration.

In the rest of the instructions, we assume the container was started using:

```bash
docker run -it --gpus all --shm-size=2.5g --name phi image_with_pytorch_2.6_and_cuda_functional
```

Be sure the container is started and up to date:

```bash
docker start phi
docker exec -u root -it phi /bin/bash
apt update && apt upgrade -y
exit
```

---

## 📦 Deploying the Server Package

```bash
docker exec -it phi /bin/bash

python3 -m pip install --upgrade pip setuptools wheel

# ⚠️ Note: We install `setuptools` and `wheel` to support building `flash-attn` from source. 
# It may fail if your PyTorch and CUDA versions don't match what `flash-attn` expects.

git clone https://github.com/gh9869827/fifo-tool-airlock-model-env.git
cd fifo-tool-airlock-model-env

python3 -m pip install -e .[server]
exit
```

---

## 📥 Download Models (Phi-4 Only)

```bash
docker exec -it phi /bin/bash

huggingface-cli login
huggingface-cli download microsoft/Phi-4-mini-instruct
huggingface-cli download microsoft/Phi-4-multimodal-instruct

exit
```

---

## 🔒 Isolate the Container

This must be run *outside* the container. It disconnects the container from all networks:

```bash
docker network disconnect bridge phi
```

---

## ⚙️ Create Configuration and Start the Server

```bash
docker exec -it phi /bin/bash

# Confirm the container is isolated from the network
wget http://example.com  # Expected to fail

cd ~/fifo-tool-airlock-model-env
mv model_config.example.json model_config.json
# Edit configuration as needed

uvicorn fifo_tool_airlock_model_env.server.fastapi_server:app --host 127.0.0.1 --port 8000
```

## 🧱 Run the Bridge on the Host

> 💡**Recommended**
>
> Activate a Python virtual environment before installing the bridge.  
> This keeps your global Python environment clean and avoids dependency conflicts.
>
> If you don't already have one:
>
> ```bash
> python3 -m venv airlock-bridge-env
> source airlock-bridge-env/bin/activate
> ```

Then:

```bash
git clone https://github.com/gh9869827/fifo-tool-airlock-model-env.git

cd fifo-tool-airlock-model-env

python3 -m pip install -e .[bridge]

uvicorn fifo_tool_airlock_model_env.bridge.fastapi_server:app --host 127.0.0.1 --port 8000
```

## Run an example

```bash
cd fifo_tool_airlock_model_env/examples
python call_model.py
```

---

## 🔒 SSL & Localhost Security

This project focuses on **isolating the model from the internet** to safely run potentially untrusted `trust_remote_code=True` models  
within a no-network environment. It assumes the host is **not shared** and is fully under your control.  
The system communicates exclusively over `localhost` and through `stdin`/`stdout` pipes.  
This local-only communication is currently **not encrypted** with SSL/TLS as part of this project.

If encryption is required, secure the `uvicorn` servers with SSL or a reverse proxy,  
and encrypt the `stdin`/`stdout` channel between the client and the bridge.

⚠️ Never bind this server to a non-`localhost` interface without proper SSL/TLS configuration.  
If you need remote access, keep the bridge bound to `localhost` and use **SSH tunneling** (`ssh -L`) for secure communication.

---

## 📦 Goals
- Minimal trusted surface area
- Clean interface between components
- Easy to extend or audit

---

## ✅ License
MIT — see [LICENSE](LICENSE) for details.
