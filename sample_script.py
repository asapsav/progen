from jax import random
from haiku import PRNGSequence
from progen_transformer import ProGen

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
print('Hello! All dependencies are installed')
print(model)