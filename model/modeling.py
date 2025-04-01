from os import PathLike
from typing import Tuple,Dict,Optional
import torch
import math
import torch.nn as nn
from torch import Tensor
from transformers import (
    PreTrainedModel,
    GPT2LMHeadModel,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput,CausalLMOutput
from transformers.pipelines import AutoTokenizer,AutoConfig,AutoModel

from model.configuration_model import GCapconfig

from icecream import ic

class MLP(nn.Module):
    def __init__(self, config: GCapconfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.intermediate_size)
        self.fc3 = nn.Linear(config.intermediate_size, config.language_model_config.n_embd) # 2 vs 3
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x
    
class GCapAttnMLP(nn.Module):
    def __init__(self, config: GCapconfig) -> None:
        super().__init__()
        self.config = config
        in_features = config.hidden_size
        self.act = nn.SiLU()

        self.w1 = nn.Linear(in_features, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, in_features)
        self.w3 = nn.Linear(in_features, config.intermediate_size)
        self.ffn_ln = nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act(self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states

class GCapMHA(nn.Module):
    def __init__(self, config: GCapconfig):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs

class GCapCrossOutput(nn.Module):
    def __init__(self, config: GCapconfig):
        super().__init__()
        dim = config.hidden_size
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = GCapAttnMLP(config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor + self.out_proj(hidden_states)
        input_tensor = input_tensor + self.mlp(self.norm2(input_tensor))
        return input_tensor

class GCapAttention(nn.Module):
    def __init__(self, config: GCapconfig):
        super().__init__()
        self.attention = GCapMHA(config)
        self.output = GCapCrossOutput(config)
        self.pruned_heads = set()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.normk = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # HACK we apply norm on q and k
        hidden_states = self.norm1(hidden_states)
        encoder_hidden_states = self.normk(encoder_hidden_states)
        encoder_hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        if attention_mask and encoder_attention_mask:
            encoder_attention_mask = torch.cat([attention_mask, encoder_attention_mask], dim=-1)
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class GCapLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.seq_len_dim = 1

        self.layer_idx = layer_idx

        self.selfattention = GCapAttention(config)
        self.has_cross_attention = True

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be given for cross-attention layers")
        self_attention_outputs = self.selfattention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions=output_attentions,
        )
        self_attention_output = self_attention_outputs[0]

        outputs = (self_attention_output,)
        return outputs

class GCapEncoder(nn.Module):
    def __init__(self, config: GCapconfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [GCapLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layers[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]

        return ModelOutput(
            last_hidden_state=hidden_states,
        )

class GCap(PreTrainedModel):
    """
    Model definition here

    Input:
        Contains input data, label
    Output:
        Inherit from ModelOutput -> the first element must be loss

    """
    config_class = GCapconfig

    def __init__(
        self, 
        config: GCapconfig, 
        *inputs, 
        **kwargs
        ):
        super(GCap,self).__init__(config, *inputs, **kwargs)
        self.config = config

        self.crossattn = GCapEncoder(config)
        self.mlp = MLP(config)
        self.language_model = GPT2LMHeadModel(config.language_model_config)

        self.loss = nn.CrossEntropyLoss()
        
    def get_text_embeddings(self,):
        return self.language_model.get_input_embeddings()
    
    def get_ltor_masks_and_position_ids_from_embeddings(self, data):
        """Build masks and position id for left to right model."""

        # Extract batch size and sequence length.
        micro_batch_size, seq_length = data.size()[:2]

        # Attention mask (lower triangular).
        att_mask_batch = 1
        attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length
        )

        # Convert attention mask to binary:
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
        attention_mask = attention_mask.masked_fill(attention_mask == 1, 0)

        return attention_mask

    def get_media_indices(self, my_list):
        if isinstance(my_list, torch.Tensor):
            my_list = my_list.cpu().tolist()
        result = []
        for i in range(len(my_list)):
            if i == 0 and my_list[i] < 0:
                result.append(i)
            elif my_list[i] != my_list[i - 1] and my_list[i] < 0:
                result.append(i)
        return result

    def forward(
            self,
            text_feature: torch.FloatTensor,
            text_neighbors: torch.FloatTensor,
            input_ids: torch.LongTensor,
            label_mask: torch.LongTensor,
            non_padding_mask: torch.LongTensor,
            non_media_mask: torch.LongTensor,
            *args,
            **kwargs,
        ) -> CausalLMOutput:

        text_tokens_ = input_ids.clone()
        batch_size = text_tokens_.size(0)

        # transform to align
        fused = self.crossattn(hidden_states = text_neighbors, encoder_hidden_states = text_feature.unsqueeze(1))[0]
        fused = self.mlp(fused)
        #

        media_token_indices = [
            self.get_media_indices(text_tokens_[i][:-1])
            for i in range(batch_size)
        ]

        text_tokens_[text_tokens_ < 0] = self.config.pad_token_id
        text_seq_length = fused.size(1)

        text_chunk_embeds = []
        for b in range(batch_size):
            start = 0
            result = []
            if len(media_token_indices[b]) > 0:
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        result.append(self.get_text_embeddings()(text_tokens_[b, start:pos].clone()))
                    result.append(fused[b])
                    start = pos + text_seq_length
            if start < text_tokens_.size(1):
                result.append(self.get_text_embeddings()(text_tokens_[b, start:].clone()))

            text_chunk_embeds.append(torch.cat(result, dim=0))

            # Actual Input Embeddings
        input_embeds = torch.stack(text_chunk_embeds, dim=0)
        attention_mask = self.get_ltor_masks_and_position_ids_from_embeddings(input_embeds)

        logits = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask
        )[0]

        loss_mask = torch.ones_like(input_ids[:, 1:],dtype=torch.long)
        loss_mask = loss_mask * label_mask[:,1:] * non_media_mask * non_padding_mask
        
        text_tokens_[:,1:][loss_mask != 1] = -100

        if text_tokens_ is not None:
            shifted_logits = logits[:,:-1,:].contiguous()
            shift_labels = text_tokens_[:,1:].contiguous()
            # ic(input_ids[0])
            # ic(shift_labels[0])
            
            shifted_logits = shifted_logits.view(-1, self.config.language_model_config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(logits.device)
            loss = self.loss(shifted_logits, shift_labels)

        # ic(loss.item())
        #when computing loss, remember to distinct single batch when training or batch of list when eval

        return ModelOutput(
            loss = loss,
            logits = logits
        )
    
    @torch.no_grad()
    def generate(
        self, 
        visual_feature: torch.Tensor,
        text_features: torch.Tensor,
        **generate_kwargs, 
    ):
        batch_size = text_features.size(0)

        #
        fused = self.crossattn(hidden_states = text_features, encoder_hidden_states = visual_feature.unsqueeze(1))[0]
        fused = self.mlp(fused)
        #
        
        sep_token_embed = self.get_text_embeddings()(torch.ones((batch_size, 1),dtype=torch.long,device=text_features.device) * 25)

        inputs_embeds = torch.cat((fused, sep_token_embed.clone()),dim=1)
        generated_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds.to(self.language_model.dtype),
            **generate_kwargs
        )

        return ModelOutput(
            generated_ids = generated_ids
        )


AutoConfig.register("GCap",GCapconfig)
AutoModel.register(GCapconfig,GCap)

if __name__ == "__main__":
    pass