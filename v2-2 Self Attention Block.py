"""
Bi-gram.py - Self Attention Layer (1:00:21)
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


# Bi-gram Model:
# ===================================================
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # New: Each Position from 1, to Block Size (T) is also encoded in an embedding table n_embd.
        # Hence, each position in the block is represented by an embedding.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd) # New: Head Object defined as a layer in this model.
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
        x = tok_emb + pos_emb #pos_emb gets broadcasted to (B, T, N embd)
        x = self.sa_head(x) # New: Apply one head of self attention.

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

1:01:58

We've now added another embedding matrix. This one is of size (BLOCK_SIZE, N_EMBD). What it does, for each value from
0 up to T-1 (or can be thought of as position 1, to, T). It grabs an embedding for that position of size N_EMBD.

We then created x: which is the token_embedding + positional_embedding. That means now the final "X" depends on the
token and it's location.

Say the embedding at position[0] = [0.4, 0.5] and position[1] = [0.8, 0.6].
and that the letter "A" had an embedding of [0.2, 0,1].

We can see than A at position 0 is now [0.6, 0.6] and that A at position 1 is now [1, 0.7]. They're different!


However we're not taking full advantage of this just yet, we haven't introduced the "wei" into this model yet.
Remember, this was the masking and average of tokens with their previous we covered at the end of "gpt-dev". 

We're yet to incorporate this step.

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


