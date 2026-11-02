import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from protgen.data import ProteinDataset
from torch.utils.data import DataLoader
from protgen.data import AA, FAMILIES, IGNORE_INDEX, special_tokens

class AutoregressiveMLP(nn.Module):
    def __init__(self, vocab_size, context_size=16, embed_dim=32, hidden_dim=128):
        super().__init__()

        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(context_size * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x):
        # x: (batch_size, context_size)
        x = self.embedding(x)

        # Flatten embeddings
        x = x.reshape(x.size(0), -1)

        return self.net(x)

    
train_data = ProteinDataset(path_to_data="../../../data/train.csv")
val_data = ProteinDataset(path_to_data="../../../data/val.csv")
test_data = ProteinDataset(path_to_data="../../../data/test.csv")

print(f"Data class: {train_data}")

# Get tensors
X_train = train_data.tokens
y_train_labels = train_data.labels          # classification task
y_train_shifted = train_data.tokens_shifted # generative task

print(f"# Train data: \n Tokens shape: {X_train.shape}\n Labels (shape): {y_train_labels.shape} \n Shifted shape: {y_train_shifted.shape}")


# Inddexing
x, x_shifted, y, length = train_data[0]
print(f"# Sample 0: \nTokens: \n {x}\n Shifted: \n {x_shifted}\n Label: {y}\n Length: {length}")

# DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

print(f"vocab ({train_data.vocab_size} tokens): {train_data.vocab}")
print(f"families: {train_data.fam_to_idx}")
print(f"padding is ignored in the loss with ignore_index={IGNORE_INDEX}")

sequence = train_data.df["sequence"][0]
tokens = train_data.encode(sequence)
print()
print(f"encode / decode round-trips: {train_data.decode(tokens) == sequence}")
print(f"and from the shifted target too:  {train_data.decode(x_shifted) == sequence}")

x, x_shifted, y, length = next(iter(train_loader))
print(f"batch: x {tuple(x.shape)} {x.dtype}, x_shifted {tuple(x_shifted.shape)}, y {tuple(y.shape)}")

# Generative: one distribution over the vocabulary per position.
logits_gen = torch.randn(x.shape[0], train_data.vocab_size, train_data.max_length)
print(f"generative loss:     {F.cross_entropy(logits_gen, x_shifted, ignore_index=IGNORE_INDEX):.4f}")

# Classification: one distribution over the three families per sequence.
logits_fam = torch.randn(x.shape[0], len(FAMILIES))
print(f"classification loss: {F.cross_entropy(logits_fam, y):.4f}")


# Param setting
num_epochs = 5
vocab_size = train_data.vocab_size
context_size = 16 # 32, 64, 128
embed_dim = 32
hidden_dim = 128
pad_id = 0
bos_id = 1
eos_id = 2

# MLP training
model = AutoregressiveMLP(
    vocab_size = vocab_size, 
    context_size = context_size, 
    embed_dim = embed_dim, 
    hidden_dim = hidden_dim
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # learn rate 0.0001

for epoch in range(num_epochs):
    print(f"The {epoch} epoch")
    model.train()
    total_loss = 0

    for x, x_shifted, y, length in train_loader:
        padded_x = F.pad(x, (context_size-1, 0), value=pad_id)

        windows = padded_x.unfold(
            dimension = 1,
            size = context_size,
            step = 1
        )

        B, L, C = windows.shape

        inputs = windows.reshape(B * L, C)
        targets = x_shifted.reshape(B * L)

        outputs = model(inputs)

        loss = F.cross_entropy(
            outputs,
            targets,
            ignore_index = -1
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")

@torch.no_grad()
def generate(model, bos_id, eos_id, pad_id, max_length=128, temperature=1.0):
    model.eval()
    tokens = [bos_id]
    for _ in range(max_length):
        context = tokens[-model.context_size:]

        context = [pad_id] * (model.context_size - len(context)) + context

        x = torch.tensor([context], dtype=torch.long)

        logits = model(x)[0]

        logits[pad_id] = float("-inf")
        logits[bos_id] = float("-inf")

        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == eos_id:
            break

        tokens.append(next_token)

    return tokens[1:]

for num in range(19):
    generated = generate(model, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id, max_length=128)

    sequence = ''.join(
        vocab[i]
        for i in generated
    )
print(sequence)

vocab = {v: k for k, v in train_data.vocab.items()}
rows = []

for num in range(19):
    generated = generate(model, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id, max_length=128)
    sequence = ''.join(
        vocab[i] for i in generated
    )
    rows.append({'sequence': sequence})

pd.DataFrame(rows).to_csv('generated_sequences_test.csv', index=False, encoding='utf-8-sig')