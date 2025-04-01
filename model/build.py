import torch
import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers import CLIPModel, GPT2LMHeadModel, GPT2Tokenizer
from model.configuration_model import GCapconfig, GPT2Config
from model.modeling import GCap

from icecream import ic

def build_model(model_args, model_config: GCapconfig):
    # for training time
    model = GCap(model_config)
    model.language_model = GPT2LMHeadModel.from_pretrained(model_args.language_model, config=model_config.language_model_config)
    return model

def build_tokenizer(tokenizer_name_or_path):
    tokenizer = GPT2Tokenizer.from_pretrained(
        tokenizer_name_or_path,
        add_eos_token = True,
    )
    tokenizer.bos_token_id = 0
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def build_config(model_args, tokenizer):
    config = GCapconfig(
        language_model_config=None,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    config.language_model_config = GPT2Config.from_pretrained(model_args.language_model)

    return config

if __name__=="__main__":
    pass