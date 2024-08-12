"""
GPT2: - Reproducing the 124M parameter model.


Additional Information:
- "contiguous" function - https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch

- **config_args allows for flexible parameter passing when creating GPTConfig instances or initializing models.
It's particularly useful when you want to allow users or other parts of your code to specify only the parameters
they want to change, while using default values for the rest.
- torch.topk, (used for sample generation): https://pytorch.org/docs/stable/generated/torch.topk.html
"""

# Import Statements:
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import sys


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 'bias' is what OpenAI call the mask in the GPT2-124M model
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # Batch Size, Token Sequence, Token Embedding (n_embd)
        # Calculate the query, key, values for all heads in a batch and move head forward to be the batch.
        # nh is the "number of heads", hs is the "head size" and C (number of channels) = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Query, Keys and Values for all the heads within the multi-head attention in one array.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # Attention (materializes the (T, T) matrix for all the queries * keys.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, hs) * (B, T, nh, hs)) * Normalize
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (b, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Concatenation Operation
        # Output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')  # Approximate is now dated, but was used for GPT2.
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)  # LayerNorm: Before MLA instead of After unlike attention paper.
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length (context window)
    vocab_size: int = 50257  # number of tokens: 50,000 BPE + 256 utf-8 + 1 <|endoftext|>
    n_layer: int = 12  # number of  layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # Token Embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),  # Positional Embeddings
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # 31:07 - Implementing the forward pass to get logits
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Cannot forward sequence length of {T}, as context length is {self.config.block_size}"
        # Forward the token and position embeddings:
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # Forward the blocks of the transformer:
        for block in self.transformer.h:
            x = block(x)
        # Forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # Flatten for loss calc
        return logits, loss

    # What are class methods? https://claude.ai/chat/3129336f-ee60-4310-80e2-b9efb6bdcf17
    @classmethod
    def from_pretrained(cls, model_type):
        """
        Loads pre-trained GPT-2 model weights from huggingface
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head, and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 250M params
            'gpt2-large': dict(n_layer=24, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1550M params
        }[model_type]
        config_args['vocab_size'] = 50257  # Identical for all GPT2 models
        config_args['block_size'] = 1024  # Identical for all GPT2 models

        # Create our from-scratch initialized GPT model:
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard the mask in loading weights

        # Initialize huggingface/transformer model:
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy over parameters while ensuring names and shapes match:
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # Ignore mask in HuggingFace
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # Ignore mask in HuggingFace
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'] # Manual Fix
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for manual fix, where layers are not in PyTorch format (stupid Tensorflow)
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# 1:02 - Writing a DataLoader:
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt", 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        # out-of-bounds issues:
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# Trainloader:
train_loader = DataLoaderLite(B=4, T=32)

# Creating Blank Model:
model = GPT(GPTConfig())
model.to(device)

# Training Loop:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"Step {i}, loss: {loss.item()}")

sys.exit(0)  # Fancy way of breaking out of the program early.


"""
Old Code - Starter Code for Pre-trained Model: (We've now moved to training our own model)
model = GPT.from_pretrained('gpt2')  # Creating a Model using huggingface weights.
model = GPT(GPTConfig())
model.eval()
model.to(device)


# num_return_sequences = 5
# max_length = 30

# prefix tokens (Starter text as tokens):
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to(device)
"""

# 37:07 - Generation:
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)
        # take the logits of the last position
        logits = logits[:, -1, :]
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default), avoids sampling rare tokens only top 50 tokens.
        # topk_probs here becomes (5, 50) hence topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to sequence

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)



"""
Replaced Code:

# 52:00 - Training the Model (Single Batch):
# Loading Batch of the Dataset:
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1])  # Grab the correct number of tokens IN A SEQUENCE (straight line)
x = buf[:-1].view(B,T).to(device)  # Batch of Data size B, with token length of T.
y = buf[1:].view(B, T).to(device)  # Batch of Labels (shifted by one) of size B, with the next prediction for each token T.

# =================================================
"""