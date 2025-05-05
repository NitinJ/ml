import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from torch.utils.data import DataLoader, Dataset





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)





from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import tiktoken


class BaseTokenizer:
    def encode(self, text):
        raise NotImplementedError("Subclasses should implement this!")

    def decode(self, token_ids):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_pad_token(self):
        raise NotImplementedError("Subclasses should implement this!")



class TikTokenizer(BaseTokenizer):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.vocab_size = self.tokenizer.n_vocab
        self.pad_token_id = self.vocab_size  # add pad token manually
        print(f"Vocab size: {self.vocab_size}")
        
    def encode(self, text, pad=False):
        ids = self.tokenizer.encode(text)
        return ids

    def decode(self, token_ids):
        token_ids = [i for i in token_ids if i != self.pad_token_id]
        return self.tokenizer.decode(token_ids)

    def get_pad_token(self):
        return self.pad_token_id
    

class MyTokenizer(BaseTokenizer):
    def __init__(self, data):
        super().__init__()
        chars = sorted(set(data))
        self.vocab_size = len(chars)
        print(f"Vocab size: {self.vocab_size}")

        # Create a mapping from characters to indices and vice versa
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.encoder = lambda s: [char_to_idx[c] for c in s]
        self.decoder = lambda l: ''.join([idx_to_char[i] for i in l])

    def encode(self, text):
        return self.encoder(text)

    def decode(self, token_ids):
        return self.decoder(token_ids)
    
    def get_pad_token(self):
        return None


class WordPieceTokenizer(BaseTokenizer):
    def __init__(self, texts, vocab_size, max_length=None):
        super().__init__()
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.decoder = decoders.WordPiece(prefix="##")
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        
        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[NEWLINE]"])
        self.tokenizer.train_from_iterator(texts, trainer)

        self.vocab_size = self.tokenizer.get_vocab_size()
        self.max_length = max_length
        print(f"Vocab size: {self.vocab_size}")

    def encode(self, text, pad=False):
        text = text.replace('\n', ' [NEWLINE] ')
        enc = self.tokenizer.encode(text)
        if self.max_length:
            enc.truncate(max_length=self.max_length)
            if pad:
                enc.pad(length=self.max_length, pad_id=0, pad_token="[PAD]")
        return enc.ids

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        txt = self.tokenizer.decode(ids, skip_special_tokens=False)
        txt = txt.replace(" [NEWLINE] ", '\n')
        txt = txt.replace(" [NEWLINE]", '\n')
        txt = txt.replace("[NEWLINE] ", '\n')
        return txt

    def get_pad_token(self):
        return 0


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        return self.data[idx:idx + self.block_size], \
            self.data[idx + 1:idx + self.block_size + 1]





with open("data/output_chat.txt") as f:
# with open("data/tiny_shakespeare.txt") as f:
# with open("data/ghazals/ghazals.txt") as f:
    data = f.read()

data = data[:int(len(data) * 0.75)]
print(f"Length of dataset in characters: {len(data)}")





# tokenizer = TransformerTokenizer(model_name="gpt2")
# tokenizer = TikTokenizer(model_name="gpt2", max_length=block_size)
lines_with_special_newline = [line + "[NEWLINE]" for line in data.splitlines() if line.strip()]
tokenizer = WordPieceTokenizer(lines_with_special_newline, vocab_size=2000)





# Testing the tokenizer
S = '''Thy lord is my shepherd, I shall not want.'''
# S = "A"
encoded = tokenizer.encode(S)
print(encoded, len(encoded))
print(tokenizer.decode(encoded))





# Hyperparameter
block_size = 256  # Length of each sequence

encoded_data = tokenizer.encode(data, pad=False)
print(f"Length of encoded data: {len(encoded_data)}")
print(f"Compression ratio: {len(encoded_data)*100 / len(data):.2f}%")

n1 = int(len(encoded_data) * 0.8)
n2 = int(len(encoded_data) * 0.9)

train_data  = torch.tensor(encoded_data[:n1], dtype=torch.long, device=device)
val_data    = torch.tensor(encoded_data[n1:n2], dtype=torch.long, device=device)
test_data   = torch.tensor(encoded_data[n2:], dtype=torch.long, device=device)

train_dataset   = CharDataset(train_data, block_size)
val_dataset     = CharDataset(val_data, block_size)
test_dataset    = CharDataset(test_data, block_size)

print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")





import math
from torch.optim.lr_scheduler import LambdaLR

class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim) # One-hot encoding
        self.lstm = nn.LSTM(emb_dim, emb_dim, num_layers=2, dropout=0.3, batch_first=True)
        self.ln = nn.LayerNorm(emb_dim)  # Layer normalization
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, vocab_size)
        )

    def forward(self, x, targets=None):
        emd = self.embedding(x)  # One-hot encoding
        out, _ = self.lstm(emd)  # RNN layer
        # Take the last time step's output
        # out = out[:, -1, :]
        out = self.ln(out)  # Layer normalization
        out = self.fc(out) # (B, T, C)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(out.view(-1, out.shape[-1]),
                                   targets.view(-1), 
                                   ignore_index=tokenizer.get_pad_token())
        return out, loss


@torch.no_grad()
def evaluate(model, split="val"):
    if split == "val":
        loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    else:
        loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    avg_loss = 0.0
    num_correct = 0
    for batch_idx, (x, y) in enumerate(loader):
        logits, loss = model(x, y)
        avg_loss += loss / len(loader)
        num_correct += (logits[:, -1, :].argmax(dim=1) == y[:, -1]).sum().item()
    return avg_loss.item(), num_correct*100 / len(loader.dataset)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))  # cosine decay
    return LambdaLR(optimizer, lr_lambda)


def get_last_non_pad_tokens(y, pad_token_id):
    if pad_token_id is None:
        return y[:, -1]  # fallback to last token
    mask = (y != pad_token_id).int()
    last_non_pad_indices = (mask.flip(dims=[1]).argmax(dim=1))
    last_positions = y.size(1) - 1 - last_non_pad_indices
    return y[torch.arange(y.size(0)), last_positions]





vocab_size = tokenizer.vocab_size  # Number of unique characters in the dataset
print(f"vocab_size: {vocab_size}")

# Initialize the model
model = LSTM(vocab_size=vocab_size, emb_dim=128)
model.to(device)

print(model)
print(f"Number of model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")





# TESTING THE MODEL
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Example: Iterate through the DataLoader once
for batch_idx, (x, y) in enumerate(train_loader):
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")

    print(f"Decoded x:\n{tokenizer.decode(x[0])}")
    print(f"Decoded y:\n{tokenizer.decode(y[0])}")
    
    logits, _ = model(x)
    
    print(f"Out shape = {logits.shape}")
    break





# Traning loop
epochs = 3
lr = 1e-3
batch_size = 512

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = get_cosine_schedule_with_warmup(optimizer, 100, epochs * len(train_loader))

# Define the loss function
criterion = nn.CrossEntropyLoss() if tokenizer.get_pad_token() is None \
    else nn.CrossEntropyLoss(ignore_index=tokenizer.get_pad_token())
losses = []
lossesi = []
val_losses = []

for epoch in range(epochs):
    model.train()
    val_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % (batch_size // 4) == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Train_Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            losses.append(loss.item())
            val_loss, accuracy = evaluate(model, split="val")
            val_losses.append(val_loss)
            lossesi.append(lossesi[-1] + 1 if len(lossesi) > 0 else 0)
        
    print(f"Val_Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    print("------------------------------")
    





for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.data.norm(2).item()
        print(f"Grad Norm | {name}: {grad_norm}")





import numpy as np
plt.figure(figsize=(10, 6))

# Plot losses and val_losses
plt.plot(lossesi, losses, label="Training Loss", marker='o')
plt.plot(lossesi, val_losses, label="Validation Loss", marker='x')

# Add grid lines, labels, and legend
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()





import time
def sample(n, context=""):
    model.eval()
    result = tokenizer.encode(context)[-block_size:]  # truncate
    with torch.no_grad():
        for _ in range(n):
            x = torch.tensor(result, dtype=torch.long).unsqueeze(0).to(device)
            logits, _ = model(x)
            probs = F.softmax(logits[:,-1,:], dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()
            result.append(idx)
            result = result[-block_size:]
    return tokenizer.decode(torch.tensor(result))

print(sample(100, context="Thy"))





@torch.no_grad()
def sample(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=None, device='cuda'):
    model.eval()
    
    # Encode initial context
    context_ids = tokenizer.encode(prompt)
    context = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    for _ in range(max_new_tokens):
        # Truncate to last block_size tokens
        context_cond = context[:, -block_size:]

        # Forward pass to get logits
        logits, _ = model(context_cond)
        logits = logits[:,-1,:] / temperature  # take last token's logits

        if top_k:
            v, ix = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        # Append sampled token
        context = torch.cat((context, next_id), dim=1)

    out_ids = context.squeeze().tolist()
    return tokenizer.decode(out_ids)

sample_text = sample(model, tokenizer, "E", max_new_tokens=522, temperature=0.85, top_k=10)
print(sample_text)



