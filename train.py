# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import pathlib
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from transformers.tokenization_utils import PreTrainedTokenizer

import torch
import json
import logging
import transformers

from dataset.dataset import text_dataset, text_collate
from train.model_trainner import custom_trainer
from model.modeling import GCap
from model.build import build_model,build_config,build_tokenizer
from model.configuration_model import GCapconfig

from icecream import ic

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

@dataclass
class ModelArguments:
    """
    Customizable for model Arguments
    """
    model_name_or_path: Optional[str] = field(default="")
    config_name_or_path: Optional[str] = field(default="")
    need_tokenizer: Optional[bool] = field(default=False)
    tokenizer_from_pretrained: Optional[bool] = field(default=False)
    tokenizer_name_or_path: Optional[str] = field(default="")
    tokenizer_max_length: Optional[int] = field(default="")
    language_model: Optional[str] = field(default="")
    freeze_language: Optional[bool] = field(default=False)
    version: Optional[str] = field(default="v0")

@dataclass
class DataArguments:
    """
    Customizable for train/val/test data Arguments
    """
    dataset_name: str = field(default=None,metadata={"help":"The dataset used for training, need to be resigtered in dataset"})
    data_root: str = field(default="", metadata={"help":"The root path for the data"})
    data_split: str = field(default="", metadata={"help":"The training split of the dataset"})
    k_neighbors: int = field(default="", metadata={"help":"The top-k neighbor used for training"})
    max_length: int = field(default=512)
    sigma: float = field(default=0.5)
    noise: float = field(default=0.5)
    p: float = field(default=0.5)
    neighbor_noise: float = field(default=0.5)
    percent: float = field(default=0.5)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Customizable for training argumentTrainingArguments(
    """
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    load_best_model_at_end: bool = field(default=True,
                                         metadata={"help": "If specified, make sure the frequency of eval and save should be same"})
    metric_for_best_model: str = field(default="")

    load_from_config: bool = field(default=False)
    load_from_pretrained: bool = field(default=False)

    # double_quant: bool = field(
    #     default=False,
    #     metadata={"help": "Compress the quantization statistics through double quantization."}
    # )
    # quant_type: str = field(
    #     default="nf4",
    #     metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    # )
    bits: int = field(
        default=32,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = ""
    lora_bias: str = "none"

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    ## TODO modify to custom model's module
    multimodal_keywords = ['vision_model', 'visual_abstractor']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def build_data_module(data_args, tokenizer:PreTrainedTokenizer = None,) -> Dict:
    """
    Get dataset and collator function for training
    """
    train_dataset = text_dataset(data_args, tokenizer, "train")
    eval_dataset = text_dataset(data_args, tokenizer, "val")
    collator = text_collate()
    return dict(
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = collator
    )

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)) # Auto assgin args based on class field 
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #TODO process multiple input dataset here for data_args

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # This part is prepared for future quantify
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.need_tokenizer and model_args.tokenizer_from_pretrained:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=model_args.tokenizer_max_length,
            padding_side="right",
            # use_fast=False,
        )
    else:
        tokenizer = build_tokenizer(model_args.tokenizer_name_or_path)

    if training_args.load_from_config:
        config = GCapconfig.from_pretrained(
            model_args.config_name_or_path)
    else:
        config = build_config(model_args, tokenizer)
        # config.save_pretrained(model_args.config_name_or_path+"/baseline")

    if training_args.load_from_pretrained:
        model = GCap.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    else:
        model = build_model(model_args, config)
        # model.save_pretrained(model_args.model_name_or_path+"/baseline")

    #TODO if the model need to freeze parameters
    # Please list here
    if model_args.freeze_language:
        for param in model.language_model.parameters():
            param.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable and training_args.status != "pretrain":
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model.language_model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
            
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    
    data_module = build_data_module(data_args,tokenizer,)

    trainer = custom_trainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    compute_metrics=None,
                    **data_module)

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()
    
    # TODO I dont like auto resume << REMOVE IT AND UNCOMMENT THE ABOVE CODE
    
    trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()