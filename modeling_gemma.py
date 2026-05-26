from typing import Any
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class GemmaConfig():
    def __init__(self,
                 vocab_size,                     # no. of tokens in vocabulary
                 hidden_size,                    # the size of each embedding
                 intermediate_size,              # size of the MLP/ Feed-forward layers in each block
                 num_hidden_layers,              # number of blocks in the decoder
                 num_attention_heads,            # no. of heads for Queries (part of Grouped Query Attention)
                 num_key_value_heads,            # no. of heads for Keys/Values
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
                 image_token_index = 256000,  # index of the <image> placeholder token
                 vocab_size = 257152,
                 projection_dim = 2048,       # dimension of output of linear projector (PaliGemmaMultiModalProjector)
                 hidden_size = 2048,          # dimension of Gemma deocder
                 pad_token_id = None,
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

        language_model = GemmaForCausalLM(config.textt_config) # Tranformer decoder
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def forward(self,
                input_ids: torch.LongTensor = None,            # Processed prompt from processor (including image tokens)
                pixel_values: torch.FloatTensor = None,        # Processed Images
                attention_mask: torch.Tensor | None = None,    # attention_mask from tokenizer
                kv_cache: KVCache | None = None,              
                ) -> tuple:
        assert torch.all(attention_mask==1), "The input cannot be padded"

        # 1. Extract the input text embeddings

        # The input text embeddings
        # shape: (batch_size, seq_len, hidden_size)
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
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, # resized patch embeddings
                                                                                                inputs_embeds,  # text embeddings (with placeholders for images)
                                                                                                input_ids,      # processed prompt with image tokens added
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

        