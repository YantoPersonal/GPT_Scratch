"""
Transformer - Layer Normalization (1:33:00)
by: Ieuan Griffiths

Reference Material: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1736s
"""


# Import Statements:
# ===================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


# Hyper Parameters:
# ===================================================
batch_size = 32
block_size = 8
max_iters = 3000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
torch.manual_seed(1337)
n_embd = 32


# Reading Data & Tokenization:
# ===================================================
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# (1) Generate Dictionaries for Mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# (2) Design Mapping Functions
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[l] for l in l)
# (3) Encode the Entire Dataset:
data = torch.tensor(encode(text), dtype=torch.long)
# (4) Train-Test Split:
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Data Loader & Estimate Loss (New Function):
# ===================================================
def get_batch(split, debug=False):
    data = train_data if split == 'train' else val_data
    print(f"The split used is {split}: \n{data[:100]}") if debug else None

    ix = torch.randint(len(data) - block_size, (batch_size,))
    print(f"\nThe random integers generated: {ix}") if debug else None

    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    print(f"\nSampling from the dataset sequentially from starting point(s) ix: \n{torch.stack([data[i:i + block_size] for i in ix])}") if debug else None

    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            # track loss for all batches:
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):  # New: Follows the exact code discussed in gpt-dev under "Crux of Attention".
    """ one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)  # (C, N head)
        self.query = nn.Linear(n_embd, head_size, bias = False)  # (C, N head)
        self.value = nn.Linear(n_embd, head_size, bias=False)  # (C, N head)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # (T, T)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)  # (B,T,N head)
        q = self.query(x)  # (B, T, N head)
        v = self.value(x)  # (B, T, N head)

        wei = q @ k.transpose(-2, -1) * C**0.5  # (B, T, N head) @ (B, N head, T) ==> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)

        out = wei @ v  # (B, T, T) @ (B, T, N head) ==> (B, T, N head)
        return out


class MultiHeadAttention(nn.Module): # New
    """Applies Multiple Attention layers in parallel, then concatenates the results, back down into one attention. """
    # https://arxiv.org/pdf/1706.03762 (Section 3.2.2: Multi Head Attention)

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # New: For Residual Connections: I don't understand this part. See ChatGPT conversation for explanation.
        # https://chatgpt.com/c/f2c0fdc3-3174-419d-b7a8-823ddce65762
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 4* embd due to formula in Section 3.3.
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # New: For Residual Connections, to ensure correct size for residual layer.
            # https://chatgpt.com/c/f2c0fdc3-3174-419d-b7a8-823ddce65762
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):  # New: MultiHeadAttention & FF into "block" module (diagram in paper minus cross attention).
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension (same as Token Size Embedding). n_head: the number of heads we'd like to use.
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # New: Layer Normalization - Values across a row are normalized.
        # https://arxiv.org/pdf/1607.06450

        # Note: Difference from Google Paper, we apply normalization before the layers, not after.
        # For each Token (T) of 32 numbers, are normalized at initalization.
        # gamma and beta, allow for this to change over time.
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # New: Removed manual Multi-Head + FC Layer, for repeated blocks of these layers.
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd),  # New: LayerNorm
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        :param idx: matrix of integers corresponding to characters.
        :param targets: matrix of integers corresponding to target characters.
        :return logits: associated probabilities depending on input character.
        :return loss: associated loss based on logits vs. known target.
        """
        B, T = idx.shape  # New: Grab shape, (32, 8) based on global variables.

        tok_emb = self.token_embedding_table(idx)  # (B, T, N embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, N embd)
        x = tok_emb + pos_emb
        x = self.blocks(x)

        logits = self.lm_head(x)  # (B, T, vocab_size C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)  # loss doesn't work with batch B, so flattened logits & targets.

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        :param idx: matrix of integers corresponding to characters.
        :param max_new_tokens: number of tokens to generate.
        :return: generated characters from bi-gram model.
        """
        for _ in range(max_new_tokens):
            # Crop to the last block_size tokens (context length, hence only the last T tokens are entered).
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self.forward(idx_cond)
            # Focus on only last time step
            logits = logits[:, -1, :]
            # Apply softmax for probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample:
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)


        return idx


"""
Comments: 

In the following code, we've consolidated the multi-headed attention layer, into a "block" class. Which runs this layer
followed by the fully connected layer after each attention. The FC layer aims to allow each token time to reflect on
the information it's received from the attention layer, before repeating.

There is a problem with us nesting these layers together. We're starting to face issues with how deep our network is.
Residual or "skip" connections will help ensure that gradients are getting back to weights earloer on in the network.
These connections can be seen in The Transformer - Module Architecture in the paper.
Residual Connections - https://arxiv.org/pdf/1512.03385

We've adapted the FC Layer and Multi Headed Attention, with a final linear layer at the end that 
ensures they are of size n_embd.


It feels pointless in Multi Headed Attention, and it is, you can comment out the project and the code still works,
as I thought, the data is already the right size.

"""



if __name__ == '__main__':
    model = BigramLanguageModel()
    m = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # Calculate Model Performance every eval_iteration cycle
        if iter % eval_iters == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")

        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate content from the model:
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))