import torch
import torch.nn as nn
import math

# Your custom MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    """
    Implementation of Multi-Head Attention from scratch.
    This allows the model to jointly attend to information from different 
    representation subspaces at different positions.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        # Ensure that the model dimension is divisible by the number of heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        # d_k is the dimension of each attention head's key/query/value
        self.d_k = d_model // n_heads

        # Linear layers for transforming inputs: queries, keys, values
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # Final linear layer to produce the output
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. Linearly project and reshape Q, K, V
        # Input shape: (batch_size, seq_len, d_model)
        # Reshaped shape: (batch_size, n_heads, seq_len, d_k)
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Calculate attention scores (Scaled Dot-Product Attention)
        # Q * K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. Apply the mask (if provided)
        # The mask is crucial for GPT's decoder-only architecture.
        # It prevents positions from attending to subsequent positions.
        if mask is not None:
            # The mask is broadcasted to fit the scores' shape
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # 5. Multiply weights by V to get the context vectors
        context = torch.matmul(attention_weights, V)

        # 6. Concatenate heads and apply final linear layer
        # Reshape context back to (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        return output

# Your GPTBlock class, modified to use your custom MultiHeadAttention
class GPTBlock(nn.Module):
    """
    A single block of the GPT (decoder-only) transformer.
    It consists of a masked multi-head self-attention layer followed by
    a position-wise feed-forward network. Residual connections and
    layer normalization are used around each of these two sub-layers.
    """
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        # Using your custom MultiHeadAttention class
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention sub-layer with a residual connection
        # The query, key, and value are all the same 'x' for self-attention
        attn_output = self.attention(x, x, x, mask)
        # Add & Norm: Add the output to the original input and normalize
        x = self.ln1(x + attn_output)

        # Feed-forward sub-layer with a residual connection
        ff_output = self.feed_forward(x)
        # Add & Norm
        x = self.ln2(x + ff_output)
        return x

# --- Main execution block to test the code ---
if __name__ == '__main__':
    # 1. Define Hyperparameters
    batch_size = 4      # Number of sequences to process in parallel
    seq_len = 10        # Length of each input sequence
    d_model = 512       # The dimensionality of the model's embeddings
    n_heads = 8         # Number of attention heads (must divide d_model)
    d_ff = 2048         # The dimensionality of the feed-forward layer

    # 2. Instantiate the GPT Block
    # This creates an instance of your model component
    gpt_block = GPTBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
    print("GPT Block instantiated successfully.")
    print("-" * 30)

    # 3. Create Dummy Input Data
    # In a real GPT, this would be token embeddings + positional embeddings.
    # Here, we just use random data to check the shapes and flow.
    # Shape: (batch_size, sequence_length, model_dimension)
    input_tensor = torch.randn(batch_size, seq_len, d_model)
    print(f"Shape of the input tensor: {input_tensor.shape}")
    print("-" * 30)
    
    # 4. Create the Causal Mask
    # This is essential for a GPT (decoder-only) model. It ensures that a token
    # at a given position can only attend to itself and previous tokens, not future ones.
    # `torch.triu` creates an upper triangular matrix. The mask will have `False`
    # for positions that are allowed to attend and `True` for masked-out positions.
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    # In your implementation, the mask needs to be 0 for masked and 1 for not masked
    # Let's adjust it to match the logic (mask == 0 is filled)
    causal_mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    print("Causal mask created. It prevents attending to future tokens.")
    print("Mask shape:", causal_mask.shape)
    # This will be broadcasted to (batch_size, n_heads, seq_len, seq_len)
    # inside the attention mechanism.
    print("-" * 30)

    # 5. Perform a Forward Pass
    # Pass the input tensor and the causal mask through the GPT block
    print("Performing a forward pass through the GPT block...")
    output_tensor = gpt_block(input_tensor, mask=causal_mask)
    print("Forward pass completed.")
    print("-" * 30)

    # 6. Check the Output
    # The output shape should be identical to the input shape, which is a key
    # characteristic of transformer blocks, allowing them to be stacked.
    print(f"Shape of the output tensor: {output_tensor.shape}")
    if output_tensor.shape == input_tensor.shape:
        print("Success: Output shape matches input shape.")
    else:
        print("Error: Output shape does not match input shape.")