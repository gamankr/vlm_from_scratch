import torch
import torch.nn as nn

# Paligemma comes in different sizes and each one has a different config
# So, we include a configuration class
class SiglipVisionConfig:

    def __init__(
            self,
            hidden_size=768,                  # size of embedding vector
            intermediate_size=3072,           # linear layer in FFN. Usually 3x or 4x the hidden size
            num_hidden_layers=12,             # number of layers of this vision transformer encoder 
            num_attention_heads=12,
            num_channels=3,
            image_size=224,                   # image shape of 224 x 224
            patch_size=16,                    # size of each patch (here 16x16)
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int = None,     # number of image embeddings for each image
            **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,  # e.g. 16x16
            stride=self.patch_size,       # No overlapping convolutions
            padding='valid',              # This indicates no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches    # A positional encoding for each patch
        self.position_embedding = nn.Embedding(  # Learnable embedding - map discrete tokens (like word indices or category IDs) into continuous, dense vectors
            num_embeddings=self.num_positions,   # Size of the positional encoding (how many unique positions exist).
            embedding_dim=self.embed_dim         # Size of each dense vector (the number of continuous features used to represent each position)
        )
        self.register_buffer( # If you have parameters in your model, which should be saved and restored in the "state_dict", but not trained by the 
                              # optimizer, you should register them as buffers. Buffers won’t be returned in model.parameters(), so that the optimizer 
                              # won’t have a change to update them.
            'position_ids',
            torch.arange(self.num_positions).expand((1,-1)), # Returns a new view of the self.tensor with singleton dimensions (dimensions with size 1)
                                                             # expanded to a larger size. Passing -1 as the size for a dimension means not changing the
                                                             # size of that dimension. Can also use .unsqueeze(0)
                                                             # Expaned in order to add to the patch embedding correctly                                                             
            persistent=False  # Buffers, by default, are persistent and will be saved alongside parameters. This behavior can be changed by setting 
                              # "persistent" to False. The only difference between a persistent buffer and a non-persistent buffer is that the latter 
                              # will not be a part of this module’s "state_dict".
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [batch_size, channels, height, width]

        patch_embeds = self.patch_embedding(pixel_values) # output shape - [batch_size, embed_dim, num_patches_H, num_patches_W] # [bs, 768, 14, 14] 
                                                          # where num_patches_H = height // patch_size, num_patches_W = width // patch_size

        embeddings = patch_embeds.flatten(2) # here, start_dim is 2. So the flattening starts from the third dimension
                                             # output_shape - [batch_size, embed_dim, num_patches] where num_patches = num_patches_H * num_patches_W

        embeddings = embeddings.transpose(1,2)  # Returns transpose. The given dimensions dim0 and dim1 are swapped.
                                                # [batch_size, embed_dim, num_patches] ----> [batch_size, num_patches, embed_dim]
                                                # This is done to provide the a sequence of patch_embeddings to the encoder across num_patched dimension

        embeddings = embeddings + self.position_embedding(self.position_ids) # Each positional encoding is of shape [1, self.num_patches]

        return embeddings
    

class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads # d_k from multi-head attention formula (d_k = d_model / h) (check paper section 3.2.2)
        self.scale = self.head_dim**-0.5                 # Scaling factor in scaled dot attention formula (1 / sqrt(d_k)) (check paper section 3.2.1)
        self.dropout = config.attention_dropout

        # The embeddings are transformed into 4 different representations - query, key, value, output_matrix
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self_out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # hidden_states: [batch_size, num_patches, embed_dim] 
        batch_size, seq_len, _ = hidden_states.size()
        # query states, key_states, value_states: [batch_size, num_patches, embed_dim]
        query_states = self.q_proj(hidden_states) # Q 
        key_states = self.k_proj(hidden_states)   # K
        value_states = self.v_proj(hidden_states) # V

        # embed_dim (d_model)(768) is decomposed into num_heads (h)(12) and head_dim (d_k)(64)
        # Then, the num_heads and num_patches are swapped.
        # [batch_size, num_patches, embed_dim]  ---> [batch_size, num_heads, num_patches, head_dim]
        # This is done to perform the attention across multiple heads but still have same dimensionality as a single head.
        # Each head works only with a subset of the input embedding. This is because a word has a different meaning depending
        # on the context that it appears in. In single head attention, the full embedding size of 2 tokens will used to 
        # calculate their dot product. This results in only 1 way of relating 2 tokens with each other. By splitting each 
        # token/patch into smaller embedding parts across multiple heads, we learn to relate tokens/patches with each other 
        # in multiple ways and contexts. Also, dividing the computation across multiple heads allows for the parallelizing
        # of the attention code. All heads can compute parallelly.
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # Calculate the attention using the formula Q * K^T / sqrt(d_k).
        # Q: [batch_size, num_heads, num_patches, head_dim] 
        # K^T: [batch_size, num_heads, head_dim, num_patches] - num_patches and head_dim transposed
        # Q * K^T: [batch_size, num_heads, num_patches, head_dim] * [batch_size, num_heads, head_dim, num_patches]
        #        =  [batch_size, num_heads, num_patches, num_patches] - head_dim removed during matrix multiplication
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {batch_size, self.num_heads, seq_len, seq_len}, but is"
                f" {attn_weights.size()}"
            )
        
        # Apply the softmax across the "keys" (here, columns - check Samsung Notes)
        # attn_weights: [batch_size, num_heads, num_patches_queries, num_patches_keys] - both num_patches of same size, just naming here for clarity
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training - Not used in Paligemma 
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # attn_weights: [batch_size, num_heads, num_patches_queries, num_patches_keys]
        # value_states: [batch_size, num_heads, num_patches, head_dim]
        # attn_output: [batch_size, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"àttn_output` should be of size {batch_size, self.num_heads, seq_len, self.head_dim}, but is"
                f" {attn_output.size()}"
            )
        
        # Transpose it back 
        # [batch_size, num_heads, num_patches, head_dim] ---> [batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1,2).contiguous()
        # Combine last 2 dimensions back to embed_dim - [batch_size, num_patches, embed_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # Multiply by W_o - [batch_size, num_patches, embed_dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

    
class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh') # Vision transformers train best with GELU based on experiments
        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipVisionEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: # hidden_states are the embeddings (SiglipVisionEmbeddings)
        # residual: [batch_size, num_patches, embed_dim]
        residual = hidden_states
        # No shape change
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    
    
class SiglipVisionEncoder(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipVisionEncoderLayer(config) for _ in config.num_hidden_layers])

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        # input_embeds: [batch_size, num_patches, embed_dim]
        hidden_states = input_embeds

        # No change in shape - [batch_size, num_patches, embed_dim]
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config) # Extract patches and embeddings (with positional encodings) from images
        self.encoder = SiglipVisionEncoder(config)       # Transformer Encoder that generates contextualized embeddings
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [batch_size, channels, height, width] -> [batch_size, num_patches, embedding_dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)


class SiglipVisionModel(nn.Module):
    '''
    Takes in the raw images and returns image embedding
    '''

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model =  SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [batch_size, channels, height, width] -> [batch_size, num_patches, embedding_dim]
        return self.vision_model(pixel_values=pixel_values)



