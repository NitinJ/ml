import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

# # Hyper parameters
# batch_size = 64
# block_size = 1024
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# d_embed = 768
# dropout = 0.2
# num_heads = 12
# n_layer = 12
# # Hyper parameters

# Hyper parameters for very basic testing.
batch_size = 8
block_size = 512
max_iters = 100
eval_interval = 25
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 25
d_embed = 256
dropout = 0.2
num_heads = 16
n_layer = 4
# Hyper parameters

with open('data/output_chat.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of dataset: ", len(text))
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) } # string to index
itos = { i:ch for i,ch in enumerate(chars) } # index to string
encode = lambda s: [stoi[c] for c in s] # encode a string to a list of indices
decode = lambda l: ''.join([itos[i] for i in l]) # decode a list of indices to a string

# Splits
data = torch.tensor(encode(text), dtype=torch.long)
n1 = int(len(data) * 0.8)
n2 = int(len(data) * 0.9)
train_data = data[:n1]
val_data = data[n1:n2]
test_data = data[n2:]
print(f"Length of training set: {len(train_data)}")
print(f"Length of validation set: {len(val_data)}")
print(f"Length of test set: {len(test_data)}")

# Data loading
def get_batch(split):
    ''' 
    Finds batch_size number of random sequences of length block_size from the data.
    First finds batch_size number of random locations inside data (ix)
    Then for each ix, creates a stack of numbers from ix to ix + block_size (x)
    And for each ix, creates a stack of numbers from ix + 1 to ix + block_size + 1 (y)
    '''
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Loss estimation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(d_embed, head_size, bias=False)
        self.key = nn.Linear(d_embed, head_size, bias=False)
        self.value = nn.Linear(d_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        print(f"x.shape: {x.shape}")
        # Compute query, key, value matrices
        k = self.key(x)     # (B, T, C)
        q = self.key(x)     # (B, T, C)

        print("fq,k,v .shape: {q.shape}")

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5     # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # Perform weighted self aggregation on values
        v = self.value(x)   # (B, T, C)
        out = wei @ v       # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(d_embed, 4 * d_embed),
            nn.ReLU(),
            nn.Linear(4 * d_embed, d_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.sequential(x)


class Block(nn.Module):
    def __init__(self, d_embed, num_heads):
        super().__init__()
        head_size = d_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(d_embed)
        self.ln1 = nn.LayerNorm(d_embed)
        self.ln2 = nn.LayerNorm(d_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

# language model.
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_embed)
        self.position_embedding_table = nn.Embedding(block_size, d_embed)
        self.blocks = nn.Sequential(\
            *[Block(d_embed, num_heads=num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_embed)
        self.lm_head = nn.Linear(d_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embedding = self.token_embedding_table(idx) # (B, T, C)
        pos_embedding = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_embedding + pos_embedding
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets == None:
            return logits, None

        # Cross entropy loss expects logits of shape (B, C, T) and targets of shape (B, C, T)
        # So we need to permute the logits to (B, C, T)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is of shape (B, T) and we want to generate a sequence of length T
        # We will use the last token in idx to generate the next token
        for i in range(max_new_tokens):
            # Crop idx to the last block size, otherwise the position embedding will fail.
            logits, loss = self(idx[:, -block_size:])
            # Just take the last logit (for the full blocksize)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Model creation, train and some test.
model = GPTLanguageModel()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_parameters = sum(p.numel() for p in model.parameters())
print("----------------------------------------------")
print(f"Number of parameters: {n_parameters / 1e6:.2f}M")
print("----------------------------------------------")

print("----------------------------------------------")
print("Training...")
print("----------------------------------------------")
for iter in range(max_iters):
    if iter%eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("----------------------------------------------")
print(f"Final losses: {estimate_loss()}")
print("----------------------------------------------")

# Test 
input = "I must"
n_chars = 200
print(f"Generating {n_chars} chars for input: {input}")

inputs = torch.tensor([encode(input)], device=device)
print("----------------------------------------------")
print(decode(model.generate(inputs, max_new_tokens=200).flatten().tolist()))

