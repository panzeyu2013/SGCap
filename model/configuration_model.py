import copy
from typing import Any, Dict, Union, Optional
from transformers.configuration_utils import PretrainedConfig
from transformers import GPT2Config
from icecream import ic

class GCapconfig(PretrainedConfig):
    """
    Defination here
    """
    model_type = "GCap" # model name
    is_composition = True

    def __init__(self,
        hidden_size = 512,
        intermediate_size = 2048,
        encoder_hidden_size = 512,
        num_attention_heads = 8,
        attention_probs_dropout_prob = 0.1,
        num_hidden_layers = 2,
        layer_norm_eps = 1e-6,
        bos_token_id = 0,
        pad_token_id = 0,
        language_model_config = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.encoder_hidden_size = encoder_hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_eps = layer_norm_eps
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id

        if language_model_config is None:
            language_model_config = GPT2Config().to_dict()

        self.language_model_config = GPT2Config(**language_model_config)

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.__class__.model_type
        output['language_model_config'] = self.language_model_config.to_dict()
        return output

if __name__ == "__main__":
    a = GCapconfig()
    print(a)