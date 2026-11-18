import torch
import torch.nn as nn
import torch.nn.functional as F
from protgen.data import IGNORE_INDEX

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        output, (h_n, c_n) = self.rnn(x) # For LSTM
        result = self.fc(output)

        return result

# LSTM training
def train(model, train_loader, optimizer, config, device="cpu"):
    model.train()

    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for x, x_shifted, y, length in train_loader:
        x = x.to(device)
        x_shifted = x_shifted.to(device)

        # Forward
        outputs = model(x)

        B, L, V = outputs.shape
        targets = x_shifted.reshape(B * L)

        loss = F.cross_entropy(
            outputs.reshape(B * L, V),
            targets,
            ignore_index = IGNORE_INDEX
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate loss and accuracy
        mask = x_shifted != IGNORE_INDEX

        valid_tokens = mask.sum().item()
        total_loss += loss.item() * valid_tokens

        predictions = outputs.argmax(dim=-1)
        total_correct += ((predictions == x_shifted) & mask).sum().item()

        total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    avg_accuracy = total_correct / total_tokens

    return avg_loss, avg_accuracy

# LSTM validation
@torch.no_grad()
def validation(model, val_data, optimizer, config, device="cpu"):
    model.eval()
    
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for x, x_shifted, y, length in val_data:
        x = x.to(device)
        x_shifted = x_shifted.to(device)

        # Forward
        outputs = model(x)

        B, L, V = outputs.shape
        targets = x_shifted.reshape(B * L)

        loss = F.cross_entropy(
            outputs.reshape(B * L, V),
            targets,
            ignore_index = IGNORE_INDEX
        )

        # calculate loss and accuracy
        mask = x_shifted != IGNORE_INDEX

        valid_tokens = mask.sum().item()
        total_loss += loss.item() * valid_tokens

        predictions = outputs.argmax(dim=-1)
        total_correct += ((predictions == x_shifted) & mask).sum().item()

        total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    avg_accuracy = total_correct / total_tokens

    return avg_loss, avg_accuracy

@torch.no_grad()
def generate(model, config, device, temperature=1.0):
    model.eval()
    tokens = [config["bos_id"]]
    for _ in range(config["max_length"]):
        x = torch.tensor([tokens], dtype=torch.long, device=device)

        logits = model(x)

        logits = logits[0, -1, :].clone()

        logits[config["pad_id"]] = float("-inf")
        logits[config["bos_id"]] = float("-inf")

        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == config["eos_id"]:
            break

        tokens.append(next_token)

    return tokens[1:]