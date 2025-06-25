[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

# Airlock Model Environment

A more secure, fully isolated environment for running Hugging Face models that require `trust_remote_code=True`. This architecture ensures maximum containment using a network-isolated Docker container (loopback only), communicating only through stdin/stdout with a local FastAPI bridge.

> ⚠️ **Note**: This setup is not intended to be scalable but provide a more secure way to run models requiring `trust_remote_code=True` by restricting network access inside the container where the model runs.
The SDK communicates with the container via `docker exec`, using `stdin` and `stdout` for transport

This project currently focuses on Phi4 models: they are lightweight, can be run on consumer GPU, capable for agentic deployments and currently requires `trust_remote_code=True`.
For each base model, additional LoRA adapters can be dynamically loaded, allowing multiple variants to share a single model backbone with efficient memory usage.

---

## 📚 Table of Contents

- [⚠️ Security Warnings](#%EF%B8%8F-security-warnings)
- [🧩 Architecture Overview](#-architecture-overview)
- [📁 Directory Structure](#-directory-structure)
- [🛠️ Prerequisites](#%EF%B8%8F-prerequisites)
- [📦 Deploy the Server Package](#-deploy-the-server-package)
- [📥 Download Models (Phi-4 Only)](#-download-models-phi-4-only)
- [🔒 Isolate the Container](#-isolate-the-container)
- [⚙️ Create Configuration and Start the Server](#%EF%B8%8F-create-configuration-and-start-the-server)
- [🧱 Run the Bridge on the Host](#-run-the-bridge-on-the-host)
- [🧪 Run an Example](#-run-an-example)
- [🎯 Fine-tuning](#-fine-tuning)
- [🔒 SSL & Localhost Security](#-ssl--localhost-security)
- [🎯 Goals](#-goals)
- [✅ License](#-license)
- [📄 Disclaimer](#-disclaimer)
- [📄 Attribution](#-attribution)
---

## ⚠️ Security Warnings

This project is designed to reduce the risks of running `trust_remote_code=True` models, but it does **not** eliminate them entirely. Consider the following:

- **No network isolation is absolute**  
  While this project isolates the model in a network-isolated Docker container (loopback only), vulnerabilities in Docker, the Linux kernel, or misconfigurations could potentially allow container escape or network access. Always keep your system and Docker runtime up to date.

- **Assumes a trusted host**  
  This setup assumes the host machine is not shared (single-user host) and is fully under your control. If the host is compromised, container isolation may be bypassed.

- **No SSL/TLS by default**  
  Communication between components (`bridge`, `server`, `client`) is not encrypted by default. If you need to protect data in transit — even on `localhost` — add SSL/TLS or use SSH tunneling.

- **Do not expose to public networks**  
  Never bind the FastAPI servers to non-`localhost` interfaces unless you have properly configured SSL/TLS and authentication. Exposing these endpoints could allow remote code execution.

- **Untrusted code caveat**  
  Models loaded with `trust_remote_code=True` can execute arbitrary Python code. This project reduces the risk but cannot guarantee protection from all exploits.

- **No sandboxing beyond Docker**  
  This project does not use additional sandboxing tools like `seccomp`, `AppArmor`, or `gVisor`. For higher assurance, consider adding OS-level hardening.

- **Review model code when possible**  
  Isolation helps, but it’s not a substitute for inspection. Review the code of any model you plan to run, even in a sandboxed environment.

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
├── fine_tuning/                                 # Fine-tuning utilities for supported models
│   └── phi_4/
│       ├── fine_tune.py                         # Custom fine-tuning script (adapted from sample below)
│       └── phi_microsoft/                       # Original files from Microsoft's Phi GitHub repo (kept for traceability)
│           ├── LICENSE.phi_microsoft
│           └── sample_finetune.py
├── sdk/                                         # Client SDK for interacting with bridge
│   ├── __init__.py
│   └── client_sdk.py
├── server/                                      # Airlocked model server logic (container-side)
│   ├── __init__.py                              #   Package initializer
│   ├── fastapi_server.py                        #   Launches the container-side FastAPI app
│   ├── logging_config.py                        #   Logging configuration for Uvicorn
│   ├── llm_model_loader.py                      #   Entrypoint to load models from config
│   ├── llm_model.py                             #   Abstract base class for model implementations
│   ├── llm_model_phi_4_base_with_adapters.py    #   Adds LoRA adapter handling to base loader
│   ├── llm_model_phi_4_mini_instruct.py         #   Loader for Phi-4 Mini Instruct with adapter support
│   └── llm_model_phi_4_multimodal_instruct.py   #   Loader for Phi-4 Multimodal with adapter support
├── .gitignore                                   # Git ignore rules
├── model_config.example.json                    # Config template
├── pyproject.toml                               # Build and dependency metadata
├── LICENSE                                      # MIT license
└── README.md                                    # You're here!
```

---
## 🛠️ Prerequisites

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

## 📦 Deploy the Server Package

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

This must be run *outside* the container. It disconnects the container from all external Docker networks:

```bash
# Disconnect the container from the default Docker bridge network
docker network disconnect bridge phi

# Verify that the container is not connected to any Docker-managed networks.
# It should output '{}'
docker inspect -f "{{json .NetworkSettings.Networks}}" phi
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

## 🧪 Run an example

```bash
cd fifo_tool_airlock_model_env/examples
python call_model.py
```

---

## 🎯 Fine-tuning

This project includes support for **safe, containerized fine-tuning** of `trust_remote_code=True` models using Hugging Face's [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer) and PEFT-based LoRA adapters.

The script is based on the official [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) fine-tuning example, adapted for secure execution inside the airlocked container and extended to support [fifo_tool_datasets](https://github.com/gh9869827/fifo-tool-datasets) adapters.

### 🔧 Available adapters

You can fine-tune on any supported format via the `--adapter` flag:

- `sqna` (single-turn): prompt-response pairs  
- `conversation` (multi-turn): role-tagged chat sessions  
- `dsl` (structured): system → input → DSL output triplets

### 🚀 Run the fine-tuning script inside the container

> 🛑 The following steps **must be run inside** the isolated container (e.g., `phi`):

#### 1. Install the fine-tuning package (one-time setup)

```bash
docker exec -it phi /bin/bash

cd fifo-tool-airlock-model-env

python3 -m pip install -e .[fine_tuning]

exit
```

#### 2. Fine-tune a model

```bash
docker exec -it phi /bin/bash

cd ~/fifo-tool-airlock-model-env

accelerate launch fifo_tool_airlock_model_env/fine_tuning/phi_4/fine_tune.py \
    --adapter conversation \
    --source .../custom-dataset \
    --output_dir ./checkpoint_phi4_conversation \
    --num_train_epochs 1 \
    --batch_size 4
```

### 📄 Notes

- Ensure the dataset is accessible via `datasets.load_dataset()`.

  If the container is fully offline (as per this project's isolation steps), you can use `docker cp` to copy a pre-fetched Hugging Face cache directory from a machine with internet access. This allows datasets to load without requiring network access inside the container.

- The fine-tuning script uses `trust_remote_code=True`, but this occurs **only within the airlocked container**.
- Fine-tuned checkpoints are saved to the provided `--output_dir` inside the container. They can be referenced in `model_config.json` directly to be served via the model server.
- You may adjust optimizer, LoRA config, and other hyperparameters directly in the script.

---

## 🔒 SSL & Localhost Security

This project focuses on **isolating the model from the internet** to safely run potentially untrusted `trust_remote_code=True` models 
within a network-isolated environment (loopback only). It assumes the host is **not shared** and is fully under your control. 
The system communicates exclusively over `localhost` and through `stdin`/`stdout` pipes. 
This local-only communication is currently **not encrypted** with SSL/TLS as part of this project.

If encryption is required, secure the `uvicorn` servers with SSL or a reverse proxy, 
and encrypt the `stdin`/`stdout` channel between the client and the bridge.

⚠️ Never bind this server to a non-`localhost` interface without proper SSL/TLS configuration. 
If you need remote access, keep the bridge bound to `localhost` and use **SSH tunneling** (`ssh -L`) for secure communication.

---

## 🎯 Goals
- Minimal trusted surface area
- Clean interface between components
- Easy to extend or audit

---

## ✅ License
MIT — see [LICENSE](LICENSE) for details.

---

## 📄 Disclaimer

This project is not affiliated with or endorsed by Hugging Face, FastAPI, Docker, Microsoft (Phi-4 model family), or the Python Software Foundation.  
It builds on their open-source technologies under their respective licenses.

---

## 📄 Attribution

This project includes a preserved copy of the original fine-tuning script provided by Microsoft in the [Phi-4-mini-instruct repository](https://huggingface.co/microsoft/Phi-4-mini-instruct).  
You can find it under [`fifo_tool_airlock_model_env/fine_tuning/phi_4/phi_microsoft/sample_finetune.py`](fifo_tool_airlock_model_env/fine_tuning/phi_4/phi_microsoft/sample_finetune.py), alongside the original MIT license.  
An adapted version of this script is provided in [`fifo_tool_airlock_model_env/fine_tuning/phi_4/fine_tune.py`](fifo_tool_airlock_model_env/fine_tuning/phi_4/fine_tune.py).

The usage examples of the [Phi-4 Multimodal Instruct model](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) published by Microsoft 
on Hugging Face were originally used as a starting point for developing the Phi-4 model adapter support in this project.  
This includes inspiration from both the model card and associated example code files, which are published under the MIT license and permit reuse and adaptation with attribution.
