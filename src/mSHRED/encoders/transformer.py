import torch                    #type: ignore
import torch.nn as nn           #type: ignore
import torch.nn.functional as F #type: ignore
from typing import Optional     #type: ignore
import copy


# Multi-Head Attention module
class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        E_q: int,               # Embedding dimension for queries
        E_k: int,               # Embedding dimension for keys
        E_v: int,               # Embedding dimension for values
        E_total: int,           # Total embedding dimension
        n_heads: int,           # Number of attention heads
        dropout: float,
        bias: bool,
        dtype: torch.dtype,
        device: str = 'cpu'
    ):
        
        #initialization 
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()

        #store the parameters
        self.n_heads = n_heads
        self.dropout = dropout
        self.bias = bias

        #see if Eq, Ek, Ev have the same embedding dimension
        self._qkv_same_embed_dim = (E_q == E_k and E_k == E_v)
        if self._qkv_same_embed_dim:
            # If they are the same, we can use a single linear layer for all three
            self.packed_proj = nn.Linear(E_q, 3 * E_total, bias=bias, **factory_kwargs)
        else:
            # Otherwise, we need separate linear layers for queries, keys, and values
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)

        # Output projection layer
        E_out = E_q                                                                 # Output embedding dimension is typically the same as the query embedding dimension
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)      # Output projection layer to combine the heads

        #compute the dimension of each head
        assert E_total % n_heads == 0, "Total embedding dimension must be divisible by the number of heads"
        self.E_head = E_total // n_heads  # Dimension of each head


    def forward(
        self,
        query: torch.Tensor,    # Shape: (batch_size, seq_len_q, E_q)
        key: torch.Tensor,      # Shape: (batch_size, seq_len_k, E_k)
        value: torch.Tensor,    # Shape: (batch_size, seq_len_v, E_v)
        attn_mask = None,       # Optional attention mask
        is_causal: bool = True  # Whether to apply causal masking
    ):
        
        #1. input projection
        #if we have the same dim
        if self._qkv_same_embed_dim:                                    

            #If query, key, and value are the same tensor, we can optimize by projecting them together
            if query is key and key is value:                           
                results = self.packed_proj(query)                       # Shape: (batch_size, seq_len_q, 3 * E_total)
                query, key, value = torch.chunk(results, 3, dim=-1)     # Split into three tensors for queries, keys, and values

            #If they are different tensors, we still project them together but separately
            else:
                q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)  # Split the weights into three parts for queries, keys, and values
                # If bias is used, we also need to split the bias into three parts
                if self.bias:   
                    q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)        # Split the bias into three parts for queries, keys, and values
                else:
                    q_bias, k_bias, v_bias = None, None, None

                # Project queries, keys, and values using the respective weights and biases
                query = F.linear(query, q_weight, q_bias)  # Project queries using the first part of the weights and bias
                key = F.linear(key, k_weight, k_bias)        # Project keys using the second part of the weights and bias
                value = F.linear(value, v_weight, v_bias)    # Project values using the third part of the weights and bias

        # If the embedding dimensions for queries, keys, and values are different, we use separate linear layers for each
        else:
            query = self.q_proj(query)  # Project queries using the query projection layer
            key = self.k_proj(key)      # Project keys using the key projection layer
            value = self.v_proj(value)  # Project values using the value projection layer


        #2. Split the heads
        #(N,*, n_heads, E_head) -> (N, n_heads,*, E_head)
        query = query.unflatten(-1, (self.n_heads, self.E_head)).transpose(1, 2)  # Reshape and permute to separate heads
        key   =   key.unflatten(-1, (self.n_heads, self.E_head)).transpose(1, 2)  # Reshape and permute to separate heads
        value = value.unflatten(-1, (self.n_heads, self.E_head)).transpose(1, 2)  # Reshape and permute to separate heads

        #3. Scaled dot-product attention (SDPA)
        # Compute attention output using scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            query, key, value, 
            attn_mask=attn_mask, dropout_p=(self.dropout if self.training else 0.0), is_causal=is_causal
        )  
        # Reshape and permute back to the original format
        # (N, n_heads,*, E_head) -> (N,*, n_heads, E_head) -> (N,*, E_total)
        attn_out = attn_out.transpose(1, 2).contiguous().flatten(-2)  

        #4. Output projection
        # Project the concatenated attention outputs back to the original embedding dimension
        output = self.out_proj(attn_out)  # Shape: (batch_size, seq_len_q, E_out)
        return output
    
# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,           # Embedding dimension of the model
        n_heads: int,           # Number of attention heads
        dim_feedforward: int,   # Dimension of the feedforward network
        dropout: float, 
        activation: nn.Module,  # Activation function for the feedforward network
        layer_norm_eps: float,  # Epsilon for layer normalization
        norm_first: bool,       # Whether to apply layer normalization before the attention and feedforward layers
        bias: bool,
        dtype: torch.dtype,
        device: str = 'cpu'
    ):
        
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()

        # Multi-head attention layer
        self.self_attn = MultiHeadAttention(
            E_q=d_model, E_k=d_model, E_v=d_model, E_total=d_model, 
            n_heads=n_heads, dropout=dropout, bias=bias, **factory_kwargs
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)     # First linear layer of the feedforward network
        self.dropout = nn.Dropout(dropout)                                                  # Dropout layer for regularization
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)     # Second linear layer of the feedforward network

        # Normalization layers
        self.norm_first = norm_first                                                # Whether to apply layer normalization before the attention and feedforward layers
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)    # Layer normalization for the attention output
        self.dropout1 = nn.Dropout(dropout)                                         # Dropout layer for the attention output
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)    # Layer normalization for the feedforward
        self.dropout2 = nn.Dropout(dropout)                                         # Dropout layer for the feedforward output

        # Activation function for the feedforward network
        self.activation = activation

    def _self_attention_block(self, x, attn_mask, is_causal):
        # Apply multi-head self-attention
        attn_output = self.self_attn(x, x, x, attn_mask=attn_mask, is_causal=is_causal)  
        return self.dropout1(attn_output)  
    
    def _feedforward_block(self, x):
        # Apply feedforward network
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(ff_output)  
        
    def forward(self, src, src_mask=None, is_causal=True):
        
        x=src

        #if norm_first is True, we apply layer normalization before the attention and feedforward blocks
        if self.norm_first:
            x = x + self._self_attention_block(self.norm1(x), attn_mask=src_mask, is_causal=is_causal)  
            x = x + self._feedforward_block(self.norm2(x)) 

        #if norm_first is False, we apply layer normalization after the attention and feedforward blocks
        else:
            out_1 = self._self_attention_block(x, attn_mask=src_mask, is_causal=is_causal)

            #if the output of the attention block is 4D, we need to reshape it back to 3D before adding it to the input
            if out_1.dim() == 4:
                out_2 = out_1 + x.unsqueeze(1).expand_as(out_1)
            else:
                out_2 = out_1 + x

            #apply layer normalization after the attention block
            x = self.norm1(out_2)
            x = self.norm2(x + self._feedforward_block(x)) 

        #return the output of the transformer encoder layer
        return x
    
# Transformer Encoder
class TransformerEncoder(nn.Module):

    def __init__(
        self,
        encoder_layer: nn.Module,   # A single transformer encoder layer to be stacked
        num_layers: int,            # Number of transformer encoder layers to stack
        norm: Optional[nn.Module],  # Optional layer normalization to apply after the last encoder layer
        dtype: torch.dtype,
        device: str = 'cpu'
    ):
        
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.num_layers = num_layers  # Store the number of layers

        # Create a list of encoder layers by deep copying the provided encoder layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

        # Optional layer normalization to apply after the last encoder layer
        self.norm = norm


    def forward(self, src, mask=None, is_causal=True):
        
        x = src

        # Pass the input through each encoder layer in sequence
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=is_causal)

        # If a normalization layer is provided, apply it to the output of the last encoder layer
        if self.norm is not None:
            x = self.norm(x)

        # Return the output of the transformer encoder
        return x

#Positional Encoding
class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,           # Embedding dimension of the model
        seq_len: int,           # Maximum sequence length for positional encoding
        dropout: float,         # Dropout rate for regularization
        device: str = 'cpu',    # Device to run the model on (e.g., "cpu" or "cuda")
    ):
        
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)                  # Dropout layer for regularization

        #Encodings
        pos_encoding = torch.zeros(1, seq_len, d_model)                                                             # Initialize a tensor to hold the positional encodings
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)                                         # Create a tensor of positions (0 to seq_len-1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))   # Compute the scaling factor for the sine and cosine functions
        
        # Apply sine to even indices in the embedding dimension
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)  
        # Apply cosine to odd indices in the embedding dimension
        if d_model % 2 == 1:
            pos_encoding[0, :, 1::2] = torch.cos(position * div_term[:-1])  #(if d_model is odd)
        else:
            pos_encoding[0, :, 1::2] = torch.cos(position * div_term)  #(if d_model is even)
        
        # Register the positional encoding
        self.register_buffer('pe', pos_encoding)  

    def forward(self, x):
        # Add positional encoding to the input and apply dropout
        x = x + self.pe[:, :x.size(1), :]   # Add positional encoding to the input
        return self.dropout(x)              # Apply dropout and return the result

# Transformer
class Transformer(nn.Module):

    def __init__(
        self,
        d_model: int,               # Embedding dimension of the model
        n_heads: int,               # Number of attention heads
        num_encoder_layers: int,    # Number of transformer encoder layers to stack
        dim_feedforward: int,       # Dimension of the feedforward network in each encoder layer
        dropout: float,             # Dropout rate for regularization
        activation: nn.Module,      # Activation function for the feedforward network in each encoder layer
        layer_norm_eps: float,      # Epsilon for layer normalization in each encoder layer
        norm_first: bool,           # Whether to apply layer normalization before the attention and feedforward layers in each encoder layer
        bias: bool,                 # Whether to include bias terms in the linear layers
        input_length: int,          # Maximum input sequence length for positional encoding
        hidden_size: int,           # Hidden size for the input projection layer
        device: str = "cpu",        # Device to run the model on (e.g., "cpu" or "cuda")
        double_channels: bool = False, # Whether to double the input channels by concatenating the mask
        **kwargs,
    ):
        
        super().__init__()
        dtype=torch.float32
        self.double_channels = double_channels  # Whether to double the input channels by concatenating the mask

        # Input projection layer to map input features to the model's embedding dimension
        # If double_channels is True, the input dimension is doubled because the mask is concatenated to src.
        self.input_embedding = nn.Linear(
            d_model * 2 if double_channels else d_model, hidden_size, bias=bias, device=device
        )  
        # Encoder
        self.encoder = TransformerEncoderLayer(
            d_model=hidden_size, n_heads=n_heads, dim_feedforward=dim_feedforward, 
            dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps,
            norm_first=norm_first, bias=bias, device=device, dtype=dtype
        )
        # Encoder Norm
        encoder_norm = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps, device=device
        )  
        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=self.encoder, num_layers=num_encoder_layers, 
            norm=encoder_norm, device=device, dtype=dtype
        )
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_size, seq_len=input_length+10, 
            dropout=dropout, device=device
        )

    def forward(self, src, src_mask=None, is_causal=True):

        # src_mask is provided in sensor space (batch, lags, sensors).
        # Apply masking before embedding so dimensions remain consistent.
        if src_mask is None:
            src_mask = torch.ones_like(src)

        if self.double_channels:
            src = torch.cat([src, src_mask], dim=-1)  # append mask channel

        # Embed the input and add positional encoding
        src = self.input_embedding(src)  # Project input features to the model's embedding dimension
        src = self.pos_encoder(src)      # Add positional encoding to the input

        # Pass the input through the transformer encoder
        transformer_output = self.transformer_encoder(
            src, mask=None, is_causal=is_causal
        ) 

        # Return only the last timestep's output for each sequence in the batch
        # SHAPE: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
        transformer_output = transformer_output[:, -1, :]  # Extract the output corresponding to the last timestep for each sequence

        assert transformer_output.dim() == 2, "Expected output shape (batch_size, hidden_size), got {}".format(transformer_output.shape)

        # Return the output of the transformer encoder
        return transformer_output
