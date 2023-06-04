from jax import random
from haiku import PRNGSequence
from progen_transformer import ProGen
import jax

model = ProGen(
    num_tokens = 256,
    dim = 512,
    seq_len = 1024,
    window_size = 256,       # local attention window size
    depth = 12,              # depth
    heads = 8,               # attention heads
    dim_head = 64,           # dimension per head
    ff_glu = True,           # use GLU in feedforward, from Noam's paper
    global_mlp_depth = 2     # last N global gmlp layers
)

rng = PRNGSequence(42)
seq = random.randint(next(rng), (1024,), 0, 256)

params = model.init(next(rng), seq)
logits = model.apply(params, next(rng), seq) # (1024, 256)
print('Hello! ProGen model:')
print(model)
# Define your model parameters
model_params = {
    'num_tokens': 10000,  # Number of tokens in your vocabulary
    'dim': 512,  # Dimension of the model
    'seq_len': 256,  # Length of the sequence
    'depth': 6,  # Number of layers in the model
    'window_size': 256,  # Size of the attention window
    'global_mlp_depth': 2,  # Depth of the global MLP
    'heads': 8,  # Number of attention heads
    'dim_head': 64,  # Dimension of each attention head
    'ff_mult': 4,  # Multiplier for the feed-forward network dimension
    'ff_glu': True,  # Whether to use GLU in the feed-forward network
    'attn_dim': None,  # Dimension of the attention mechanism
    'clamp_gate': True,  # Whether to clamp the gate
    'shift_tokens': True  # Whether to shift tokens
}

# Create an instance of the model
model = ProGen(**model_params)

# Initialize the model with some random input
rng = jax.random.PRNGKey(42)
input_seq = jax.random.randint(rng, (256,), 0, 10000)  # Random sequence of length 256
params = model.init(rng, input_seq)

# Flatten the parameters
params_flat, _ = jax.tree_flatten(params)

# Print the shapes
print('The shapes')
for param in params_flat:
    print(param.shape)
