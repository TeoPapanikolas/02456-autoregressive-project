import torch
import torch.nn as nn
import torch.nn.functional as F
from protgen.data import IGNORE_INDEX

class AutoregressiveMLP(nn.Module):
    def __init__(self, vocab_size, context_size, embed_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.Flatten(),
            nn.Linear(context_size * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x):
        return self.net(x)

# MLP training
def train(model, train_loader, optimizer, config, device="cpu"):
    model.train()
    total_loss = 0

    for x, x_shifted, y, length in train_loader:
        x = x.to(device)
        x_shifted = x_shifted.to(device)
    
        padded_x = F.pad(x, (config["context_size"]-1, 0), value=config["pad_id"])

        windows = padded_x.unfold(
            dimension = 1,
            size = config["context_size"],
            step = 1
        )

        # B: batch_size; L: length; C: context_size
        B, L, C = windows.shape

        inputs = windows.reshape(B * L, C)
        targets = x_shifted.reshape(B * L)

        outputs = model(inputs)

        loss = F.cross_entropy(
            outputs,
            targets,
            ignore_index = IGNORE_INDEX
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    return avg_loss, 0

# MLP validation
@torch.no_grad()
def validation(model, val_data, optimizer, config, device="cpu"):
    model.eval()
    val_loss = 0
    for x, x_shifted, y, length in val_data:
        x = x.to(device)
        x_shifted = x_shifted.to(device)
    
        padded_x = F.pad(x, (config["context_size"]-1, 0), value=config["pad_id"])
        windows = padded_x.unfold(dimension=1, size=config["context_size"], step=1)
        B, L, C = windows.shape
        inputs = windows.reshape(B * L, C)
        targets = x_shifted.reshape(B * L)
        outputs = model(inputs)

        val_loss += F.cross_entropy(outputs, targets, ignore_index=IGNORE_INDEX).item()
    
    avg_loss = val_loss / len(val_data)

    return avg_loss, 0

# MLP generate
@torch.no_grad()
def generate(model, config, device, temperature=1.0):
    model.eval()
    tokens = [config["bos_id"]]
    for _ in range(config["max_length"]):
        context = tokens[-model.context_size:]

        context = [config["pad_id"]] * (model.context_size - len(context)) + context

        x = torch.tensor([context], dtype=torch.long, device=device)

        logits = model(x)[0]

        logits[config["pad_id"]] = float("-inf")
        logits[config["bos_id"]] = float("-inf")

        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == config["eos_id"]:
            break

        tokens.append(next_token)

    return tokens[1:]