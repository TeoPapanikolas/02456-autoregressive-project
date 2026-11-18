import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from protgen.data import ProteinDataset
from torch.utils.data import DataLoader
from protgen.data import AA, FAMILIES, IGNORE_INDEX, special_tokens
import random
from torch.nn.utils.rnn import pad_sequence


class NoisyPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, length):
        x = self.embedding(x)
        output, (h_n, c_n) = self.rnn(x) # For LSTM

        idx = torch.arange(x.size(0), device=x.device)
        last_hidden = output[idx, length - 1]
        result = self.fc(last_hidden)

        return result

def make_noisy(x, lengths):
    noisy_sequences = []

    for seq, length in zip(x, lengths):
        prefix_length = random.randint(1, int(length))

        # BOS + randomly selected amino acids
        truncated = seq[:prefix_length + 1]

        noisy_sequences.append(truncated)

    new_lengths = torch.tensor(
        [len(seq) for seq in noisy_sequences],
        device=x.device
    )

    x_noisy = pad_sequence(
        noisy_sequences,
        batch_first=True,
        padding_value=0
    )

    return x_noisy, new_lengths

# NoisyPredictor training
def train(predictor, train_loader, optimizer, config, device="cpu"):
    predictor.train()

    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for x, x_shifted, y, length in train_loader:
        x_noisy, new_length = make_noisy(x, length)

        x_noisy = x_noisy.to(device)
        x_shifted = x_shifted.to(device)
        new_length = new_length.to(device)
        y = y.to(device)

        # Forward
        outputs = predictor(x_noisy, new_length)

        loss = F.cross_entropy(
            outputs,
            y
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate loss and accuracy
        total_loss += loss.item() * y.size(0)

        predictions = outputs.argmax(dim=-1)
        total_correct += ((predictions == y)).sum().item()

        total_tokens += y.size(0)

    avg_loss = total_loss / total_tokens
    avg_accuracy = total_correct / total_tokens

    return avg_loss, avg_accuracy

# NoisyPredictor validation
@torch.no_grad()
def validation(predictor, val_data, optimizer, config, device="cpu"):
    predictor.eval()
    
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for x, x_shifted, y, length in val_data:
        x_noisy, new_length = make_noisy(x, length)

        x_noisy = x_noisy.to(device)
        x_shifted = x_shifted.to(device)
        new_length = new_length.to(device)
        y = y.to(device)

        # Forward
        outputs = predictor(x_noisy, new_length)

        loss = F.cross_entropy(
            outputs,
            y
        )

        # calculate loss and accuracy
        total_loss += loss.item() * y.size(0)

        predictions = outputs.argmax(dim=-1)
        total_correct += ((predictions == y)).sum().item()

        total_tokens += y.size(0)

    avg_loss = total_loss / total_tokens
    avg_accuracy = total_correct / total_tokens

    return avg_loss, avg_accuracy

