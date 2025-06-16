# Adapted from Microsoft's Phi-4-mini-instruct sample fine-tuning script
#
# Original source:
# https://huggingface.co/microsoft/Phi-4-mini-instruct/resolve/main/sample_finetune.py
#
# Original file preserved at:
# fifo_tool_airlock_model_env/fine_tuning/phi_4/phi_microsoft/sample_finetune.py
#
# This is the customized version:
# fifo_tool_airlock_model_env/fine_tuning/phi_4/fine_tune.py
#
# Licensed under the MIT License (see LICENSE.phi_microsoft)

import sys
import logging
import argparse
from typing import Type, Any, cast

from peft import LoraConfig
import torch

# Pylance: suppress missing type stub warning for datasets
import datasets  # type: ignore
import transformers  # type: ignore
from trl import SFTTrainer  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    PreTrainedModel
)

from fifo_tool_datasets.sdk.hf_dataset_adapters.common import DatasetAdapter
from fifo_tool_datasets.sdk.hf_dataset_adapters.sqna import SQNAAdapter
from fifo_tool_datasets.sdk.hf_dataset_adapters.conversation import ConversationAdapter
from fifo_tool_datasets.sdk.hf_dataset_adapters.dsl import DSLAdapter

_ = """
A simple example on using SFTTrainer and Accelerate to finetune Phi-4-Mini-Instruct model. For
a more advanced example, please follow HF alignment-handbook/scripts/run_sft.py.
This example has utilized DeepSpeed ZeRO3 offload to reduce the memory usage. The
script can be run on V100 or later generation GPUs. Here are some suggestions on 
futher reducing memory consumption:
    - reduce batch size
    - decrease lora dimension
    - restrict lora target modules
Please follow these steps to run the script:
1. Install dependencies: 
    conda install -c conda-forge accelerate=1.3.0
    pip3 install -i https://pypi.org/simple/ bitsandbytes
    pip3 install peft==0.14.0
    pip3 install transformers==4.48.1
    pip3 install trl datasets
    pip3 install deepspeed
2. Setup accelerate and deepspeed config based on the machine used:
    accelerate config
Here is a sample config for deepspeed zero3:
    compute_environment: LOCAL_MACHINE
    debug: false
    deepspeed_config:
      gradient_accumulation_steps: 1
      offload_optimizer_device: none
      offload_param_device: none
      zero3_init_flag: true
      zero3_save_16bit_model: true
      zero_stage: 3
    distributed_type: DEEPSPEED
    downcast_bf16: 'no'
    enable_cpu_affinity: false
    machine_rank: 0
    main_training_function: main
    mixed_precision: bf16
    num_machines: 1
    num_processes: 4
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
3. check accelerate config:
    accelerate env
4. Run the code:
    accelerate launch sample_finetune.py
"""

ADAPTER_MAP: dict[str, Type[DatasetAdapter]] = {
    "sqna": SQNAAdapter,
    "conversation": ConversationAdapter,
    "dsl": DSLAdapter,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", choices=ADAPTER_MAP.keys(), required=True)
    parser.add_argument("--source", required=True, help="HF repo id")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()

def load_dataset_from_source(adapter_name: str, source: str) -> datasets.DatasetDict:
    adapter_cls = ADAPTER_MAP[adapter_name]()
    return adapter_cls.from_hub_to_dataset_dict(source)

def apply_chat_template(example: dict[str, Any],
                        tokenizer: transformers.PreTrainedTokenizerBase) -> dict[str, Any]:
    messages = example["messages"]
    # Pylance: Type of apply_chat_template() is partially unknown
    example["text"] = tokenizer.apply_chat_template(  # type: ignore[reportUnknownMemberType]
        messages, tokenize=False, add_generation_prompt=False)
    return example

def main() -> None:
    args = parse_args()

    ###################
    # Hyper-parameters
    ###################
    training_config: dict[str, Any] = {
        "bf16": True,
        "do_eval": False,
        "learning_rate": 5.0e-06,
        "log_level": "info",
        "logging_steps": 20,
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "num_train_epochs": args.num_train_epochs,
        "max_steps": -1,
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": args.batch_size,
        "per_device_train_batch_size": args.batch_size,
        "remove_unused_columns": True,
        "save_steps": 1000,
        "save_total_limit": 1,
        "seed": 0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.2,
    }

    peft_config: dict[str, Any] = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
        "modules_to_save": None,
    }

    train_conf = TrainingArguments(**training_config)
    peft_conf = LoraConfig(**peft_config)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = train_conf.get_process_log_level()
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s distributed training: %s, 16-bits training: %s",
        train_conf.local_rank,
        train_conf.device,
        train_conf.n_gpu,
        bool(train_conf.local_rank != -1),
        train_conf.fp16
    )
    logger.info("Training/evaluation parameters: %s", train_conf)
    logger.info("PEFT parameters: %s", peft_conf)

    ################
    # Model Loading
    ################
    checkpoint_path = args.model
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    # Cast is needed because return type of from_pretrained() is not declared statically
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(  # type: ignore[reportUnknownMemberType]
            checkpoint_path,
            **model_kwargs
        )
    )
    # Pylance: Type of from_pretrained() is partially unknown
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[reportUnknownMemberType]
        checkpoint_path
    )
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token  # type: ignore[reportUnknownMemberType]
    tokenizer.pad_token_id = \
        tokenizer.convert_tokens_to_ids(  # type: ignore[reportUnknownMemberType]
            tokenizer.pad_token  # type: ignore[reportUnknownMemberType]
    )
    tokenizer.padding_side = 'right'

    ##################
    # Data Processing
    ##################
    dataset_dict = load_dataset_from_source(args.adapter, args.source)
    train_dataset = dataset_dict["train"].shuffle(seed=42)
    eval_dataset = dataset_dict["validation"].shuffle(seed=51)

    # Pylance: Type of map() is partially unknown
    processed_train_dataset = train_dataset.map(  # type: ignore[reportUnknownMemberType]
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=list(train_dataset.features),
        desc="Applying chat template to train set",
    )

    # Pylance: Type of map() is partially unknown
    processed_eval_dataset = eval_dataset.map(  # type: ignore[reportUnknownMemberType]
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=list(eval_dataset.features),
        desc="Applying chat template to eval set",
    )

    ###########
    # Training
    ###########
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        peft_config=peft_conf,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_eval_dataset
    )
    # Pylance: Type of train() is partially unknown
    train_result = trainer.train()  # type: ignore[reportUnknownMemberType]
    metrics = train_result.metrics  # type: ignore[reportUnknownMemberType]
    # Pylance: Type of log_metrics() is partially unknown
    trainer.log_metrics("train", metrics)  # type: ignore[reportUnknownMemberType]
    # Pylance: Type of save_metrics() is partially unknown
    trainer.save_metrics("train", metrics)  # type: ignore[reportUnknownMemberType]
    trainer.save_state()

    #############
    # Evaluation
    #############
    tokenizer.padding_side = 'left'
    # Pylance: Type of evaluate() is partially unknown
    metrics = trainer.evaluate()  # type: ignore[reportUnknownMemberType]
    metrics["eval_samples"] = len(processed_eval_dataset)
    # Pylance: Type of log_metrics() is partially unknown
    trainer.log_metrics("eval", metrics)  # type: ignore[reportUnknownMemberType]
    # Pylance: Type of save_metrics() is partially unknown
    trainer.save_metrics("eval", metrics)  # type: ignore[reportUnknownMemberType]

    ############
    # Save model
    ############
    trainer.save_model(train_conf.output_dir)

if __name__ == "__main__":
    main()
