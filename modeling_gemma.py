from typing import Any
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache():

    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the each time in key_cache is [batch_size, num_heads_KV, seq_len, head_dim]
            return self.key_cache[0].shape[-2]
        
    def update(self,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               layer_idx: int,  # which decoder block/layer does this kv_cache belong to 
               ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this decoder layer, let's create it
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Otherwise, concatenate the new keys with the existing ones.
            # Each tensor is of shape: [batch_size, num_heads_KV, seq_len, head_dim]
            self.key_cache[layer_idx] = torch.concat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.concat([self.value_cache[layer_idx], value_states], dim=-2)

        # then return all the existing keys/values + the new ones
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GemmaConfig():
    def __init__(self,
                 vocab_size,                     # no. of tokens in vocabulary
                 hidden_size,                    # the size of each embedding
                 intermediate_size,              # size of the MLP/ Feed-forward layers in each block
                 num_hidden_layers,              # number of decoder blocks/layers in the decoder
                 num_attention_heads,            # no. of heads for Queries (part of Grouped Query Attention)
                 num_key_value_heads,            # no. of heads for Keys/Values (different for Queries due to Grouped Query Attention)
                 head_dim = 256,                 # size of dimension of each head
                 max_position_embeddings = 8192, # max no. of positions/tokens the model is trained on (max context length)
                 rms_norm_eps=1e-6,              # RMS Normalization factor
                 rope_theta = 10000.0,           # base frequency for rotary positional encoding
                 attention_bias = False,
                 attention_dropout = 0.0,
                 pad_token_id = None,
                 **kwargs,
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_prop_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():
    def __init__(self,
                 vision_config = None,        # Config of vision encoder (SigLIP)
                 text_config = None,          # Config of text decoder (Gemma)
                 ignore_index = -100, 
                 image_token_index = 256000,  # position of the <image> placeholder token in vocabulary
                 vocab_size = 257152,
                 projection_dim = 2048,       # dimension of output of linear projector (PaliGemmaMultiModalProjector)
                 hidden_size = 2048,          # dimension of embedding in Gemma decoder
                 pad_token_id = None,         # position of the padding token in vocabulary
                                              # Padding tokens are special filler symbols to make all input sequences in a batch the exact same length.
                 **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False # part of hugging face configuration
        self.pad_token_id = pad_token_id

        self.vision_config = vision_config
        self.vision_config = SiglipVisionConfig(**vision_config)

        self.text_config = text_config
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2 # No. of patches in an image
        self.vision_config.projection_dim = projection_dim

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # 1 / sqrt(...)
                                                                           # 'eps' is added to avoid division by zero
                                                                           # in case calculation results in value zero
    
    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output *= (1.0 + self.weight.float()) # gamma variable in RMS normalization is an updatable weight
        return output.type_as(x)
    
def repeat_kv(hidden_states: torch.Tensor,
              n_rep: int) -> torch.Tensor:
    batch_size, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # hidden_states[:,:,None,:,:] is for inserting dimension. Similar to hidden_states.unsqueeze(2).
    # Expand would mean the dimensions after dimension 2 (slen, head_dim) are repeated 'n_rep' times
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, slen, head_dim) # Reshaped to match the shape same as query heads


class GemmaAttention(nn.Module):
    def __init__(self, 
                 config: GemmaConfig, 
                 layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0 # make sure the full embedding can be divided into num_heads
        
        # For grouped query attention, the output sizes for linear projections are different for queries and
        # keys/values. This is unlike the Multi-head attention in SigLIP, where input and output sizes are same.
        # This is because the number of key_value_heads are less than number of query_heads (num_heads).
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias) 
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta
        )

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None,
                kv_cache: KVCache | None = None,
                **kwargs,
                ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        
        batch_size, q_len = hidden_states.size() # [batch_size, seq_len, hidden_size]
        # [batch_size, seq_len, num_heads_Q * head_dim]
        query_states = self.q_proj(hidden_states)
        # [batch_size, seq_len, num_heads_KV * head_dim]
        key_states = self.k_proj(hidden_states)
        # [batch_size, seq_len, num_heads_KV * head_dim]
        value_states = self.v_proj(hidden_states)

        #### Transpose to simulate performing computation across multiple heads
        # [batch_size, num_heads_Q, seq_len, head_dim]
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1,2)
        # [batch_size, num_heads_KV, seq_len, head_dim]
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        # [batch_size, num_heads_KV, seq_len, head_dim]
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

        #### Apply rotary positional encodings
        # [batch_size, seq_len, head_dim], [batch_size, seq_len, head_dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [batch_size, num_heads_Q, seq_len, head_dim], [batch_size, num_heads_KV, seq_len, head_dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        #### Update the kv_cache
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        #### Repeat the keys and values heads to match the number of query heads
        # So, in order to leverage Grouped Query Attention, we need to write a custom CUDA kernel which can use the optimisation
        # of using less key/value heads. But here we are only coding a naive implementation, so instead we will just copy the 
        # key/value heads to match the number of query heads. This essentially reverses the use of Grouped Query Attention here
        # and we don't get the advantage of higher throughput
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups) 

        # Calculate the attention using the formula Q * K^T / sqrt(d_k).
        # Q: [batch_size, num_heads_Q, seq_len_Q, head_dim] 
        # K^T: [batch_size, num_heads_Q, head_dim, seq_len_KV] - seq_len and head_dim transposed (KV and Q have same no. of heads after repeat_kv function above)
        # Q * K^T: [batch_size, num_heads, seq_len, head_dim] * [batch_size, num_heads, head_dim, seq_len]
        #        =  [batch_size, num_heads, seq_len, seq_len] - head_dim removed during matrix multiplication
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)

        # Attention mask is the causal mask and in Paligemma's case, it will be filled with zeros.
        # There is no masking in PaliGemma, even for the future tokens in the text prompt. This is a decision by PaliGemma authors
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Apply the softmax across the "keys" (here, columns - check Samsung Notes)
        # attn_weights: [batch_size, num_heads_Q, seq_len_Q, seq_len_kV]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training - Not used in Paligemma 
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)


        # attn_weights: [batch_size, num_heads_Q, seq_len_Q, seq_len_kV]
        # value_states: [batch_size, num_heads_KV, seq_len_KV, head_dim] # num_heads_Q and num_heads_KV is same after repeat_kv function
        # attn_output: [batch_size, num_heads_Q, seq_len_Q, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"àttn_output` should be of size {batch_size, self.num_heads, q_len, self.head_dim}, but is"
                f" {attn_output.size()}"
            )
        
        # Transpose it back 
        # [batch_size, num_heads_Q, seq_len_Q, head_dim] ---> [batch_size, seq_len_Q, num_heads_Q, head_dim]
        attn_output = attn_output.transpose(1,2).contiguous()
        # Concatenate all heads together. Combine last 2 dimensions back to hiden_size - [batch_size, seq_len_Q, hidden_size]
        attn_output = attn_output.reshape(batch_size, q_len, -1)

        # Multiply by W_o - [batch_size, num_patches, embed_dim]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
    
class GemmaMLP(nn.Module):
    '''
    Uses 3 Linear layers. Whereas the MLP in the SigLIP encoder used only 2 Linear layers
    '''
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # Equivalent to
        # y = self.gate_proj(x) # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, intermediate_size]
        # y = nn.functional.gelu(y, approximate='tanh') # [batch_size, seq_len, intermediate_size]
        # j = self.up_proj(x) # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, intermediate_size] 
        # z = y * j # [batch_size, seq_len, intermediate_size]
        # z = self.down_proj(z) # [batch_size, seq_len, intermediate_size] -> [batch_size, seq_len, hidden_size]
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate='tanh') * self.up_proj(x))

    
class GemmaDecoderLayer(nn.Module):
    '''
    Similar to encoder layer in SigLIP - SiglipVisionEncoderLayer.
    Instead of LayerNorm in SigLIP, RMS-LayerNorm is used here
    '''
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_prop_eps)
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_prop_eps)
        self.mlp = GemmaMLP(config)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None,
                kv_cache: KVCache | None = None
                ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        # [batch_size, seq_len, hidden_size]
        residual = hidden_states
        # No shape change
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache)
        hidden_states += residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual

        return hidden_states

class GemmaModel(nn.Module):
    '''
    Gemma Transformer Decoder
    '''
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # If specified, the entries at `padding_idx` in nn.Embedding do not contribute to the gradient;
        # Therefore, the embedding vector at `padding_idx` is not updated during training
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList([GemmaDecoderLayer(config, layer_idx) for layer_idx in config.num_hidden_layers])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_prop_eps)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(self,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None,
                inputs_embeds: torch.FloatTensor | None = None,
                kv_cache: KV_Cache | None = None
                ) -> torch.FloatTensor:
        # [batch_size, seq_len, hidden_size]
        hidden_states = inputs_embeds
        # [batch_size, seq_len, hidden_size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states *= normalizer

        for decoder_layer in self.layers:
            # [batch_size, seq_len, hidden_size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )
        
        # [batch_size, seq_len, hidden_size]
        hidden_states = self.norm(hidden_states)

        # [batch_size, seq_len, hidden_size]
        return hidden_states
 
class GemmaForCausalLM(nn.Module):
    '''
    Gemma Transformer Decoder + Language Modeling Head
    Language Modeling Head converts the contextualized embeddings from Gemma decoder to logits
    '''
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # Bias is False due to weight tying

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None,
                inputs_embeds: torch.FloatTensor | None = None,
                kv_cache: KVCache | None = None,
                ) -> tuple:
        
        # inputs_embeds - [batch_size, seq_len, hidden_size]
        # outputs - [batch_size, seq_len, hidden_size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return updated kv_cache
            return_data["kv_cache"] = kv_cache

        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
    '''
    Linear Projection after SigLIP Encoder that changes the embedding dimension of SigLIP encoder to the 
    embedding dimension used in the Gemma decoder
    '''
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=True)

    def forward(self, image_features):
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patched, projection_dim]
        hidden_states = self.linear(image_features)
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    '''
    It is called Conditional Generation because we are conditioning the generation of text based on input images
    '''
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config) # Transformer Encoder
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config) # Linear Projection after the Transformer Encoder
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config) # Transformer decoder
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(self,
                                        image_features: torch.Tensor,   # resized contextualized patch embeddings
                                        inputs_embeds: torch.Tensor,    # input embeddings (text embedding + placeholders for image embeddings)
                                        input_ids: torch.Tensor,        # processed prompt from tokenizer with image tokens (contains 
                                                                        # positions of tokens in vocabulary)
                                        attention_mask: torch.Tensor,   # attention mask from tokenizer (same size as input_ids filled 
                                                                        # with '1' to indicate to attend to all tokens. '0' usually indicates
                                                                        # masked tokens (tokens which should not be attended to))
                                        kv_cache: KVCache | None = None
                                        ):
        _,  _, embed_dim = image_features.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        batch_size, sequence_length = input_ids.shape

        # shape: [batch_size, seq_len, hidden_size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5) # Normalize with Gemma Decoder embedding dimension to maintain 
                                                                                # magnitude of values across patch and text embeddings
        
        # Combine the embeddings of the image tokens, text tokens and mask out all the padding tokens
        # [batch_size, seq_len, embed_dim]
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=dtype, device=device)

        # Masks for each type of token in the processed prompt. 
        # Masks are BoolTensor
        # Shape: [batch_size, seq_len]. True for text tokens and False for others
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.config.pad_token_id)
        # Shape: [batch_size, seq_len]. True for image tokens and False for others
        image_mask = input_ids == self.config.image_token_index
        # Shape; [batch_size, seq_len]. True for padding tokens and False for others
        pad_mask = input_ids == self.config.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        # [batch_size, seq_len] -> unsqueeze -> [batch_size, seq_len, 1] -> expand -> [batch_size, seq_len, embed_dim]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding) # all 3 arguments should be of same shape

        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the
        # sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)

        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        ###### CREATE THE ATTENTION MASK #######

        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1] # Sequence length (used for query, so, Query length) of input embedding (text + image embeddings)

        # prefill phase
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we are in the prefill phase
            # This only works we have no padding
            #
            # In PaliGemma, the input text tokens attend to the future tokens during attention calculation, unlike usual language models.
            # So no mask is applied. This is a technical choice as in PaliGemma's case, the input prompt is the task given by the user
            # to be done based on input image and the text tokens may benefit from the future tokens. This is PaliGemma's design choice.
            causal_mask = torch.full((batch_size, q_len, q_len),
                                    fill_value=0,
                                    dtype=dtype,
                                    device=device)
        # Decode phase (generation phase)
        else:
            # Since we are generating tokens, the query must be single token (check kv-caching technique)
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this we don't need to mask anything because when using kv-cache in decode phase, only a single query token (last 
            # token in the generated prompt) is used for attention calculation, and it needs to attend to all the previous tokens. 
            # Masks are required only when using entire Query vector of tokens (no kv-cache).
            # This only works when we have no padding
            causal_mask = torch.full((batch_size, q_len, kv_len),
                                     fill_value=0,
                                     dtype=dtype,
                                     device=device)
            
        # Add the attention head dimension
        # [batch_size, Q_len, KV_len] -> [batch_size, num_heads_Q, Q_len, KV_len]
        causal_mask = causal_mask.unsqueeze(1)


        ##### Positions of tokens used by ROTARY POSITIONAL ENCODING #####

        # Decode phase (generation phase)
        if kv_cache is not None and kv_cache.num_items() > 0: 
            # attention_mask.shape -> [batch_size, sequence_length] (same as inputs_ids) (inputs_ids and attention mask are generated by tokenizer)
            # 
            # attention_mask                   -> batch_size x [1, 1, 1,....., 1, 1, 1] - (vector is "sequence_length" times "1")
            # attention_mask.cumsum(-1)        -> batch_size x [1, 2, 3,...., sequence_length - 2, sequence_length - 1, sequence_length]
            # attention_mask.cumsum(-1)[:, -1] -> batch_size x [sequence_length]
            #
            # Thus, the position of the query is just the last position (sequence_length) for decode phase
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        # Prefill phase
        else:
            # Create a position_ids based on the size of the attention mask
            # For masked tokens, use the number 1 as position - But in PaliGemma, the tokens are not masked
            #
            # attention_mask                                                      -> batch_size x [1, 1, 1, 1, 0] (Here, last token is masked)
            # attention_mask.cumsum(-1)                                           -> batch_size x [1, 2, 3, 4, 4]
            # (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1 ) -> batch_size x [1, 2, 3, 4, 1] (Last token (masked) is replaced with 1)
            #
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids        
    
    def forward(self,
                input_ids: torch.LongTensor = None,            # processed prompt from tokenizer with image tokens (contains positions of tokens in vocabulary)
                pixel_values: torch.FloatTensor = None,        # Processed Images
                attention_mask: torch.Tensor | None = None,    # attention_mask from tokenizer
                kv_cache: KVCache | None = None,              
                ) -> tuple:
        assert torch.all(attention_mask==1), "The input cannot be padded"

        # 1. Extract the input text embeddings

        # input embeddings (text embedding + placeholders for image embeddings)
        # shape: (batch_size, seq_len, hidden_size)
        # 'self.language_model.get_input_embeddings()' returns a nn.Embedding layer to which 'input_ids' is given as input argument
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Produce contextualized patch embeddings and process them
        
        # Images -> Transformer encoder -> Patch embeddings
        # [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
        selected_image_features = self.vision_tower(pixel_values=pixel_values) 

        # Patch embeddings -> Linear Projector -> Resized patch embeddings (to the size of text embeddings)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, hidden_size]
        image_features = self.multi_modal_projector(selected_image_features)   

        # 3. Merge text and image embeddings

        # Merge the embeddings of text tokens and image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, 
                                                                                                inputs_embeds,  
                                                                                                input_ids,      
                                                                                                attention_mask,
                                                                                                kv_cache)
        
        # 4. Get the output of the decoder model

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs

        