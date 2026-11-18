import argparse
import json
import torch
from protgen.data import ProteinDataset
from torch.utils.data import DataLoader

from protgen.models import MLP, LSTM, ClassifierLSTM


def parse_args():
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--model-name", type=str, default='ClassifierLSTM')
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    # For LSTM
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()

def main():
    args = parse_args()

    train_data = ProteinDataset(path_to_data="data/train.csv")
    val_data = ProteinDataset(path_to_data="data/val.csv")
    test_data = ProteinDataset(path_to_data="data/test.csv")

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    # Param setting
    # Dataset parameters
    config = {
        "model_name": args.model_name,
        "vocab": train_data.vocab,
        "vocab_size": train_data.vocab_size,
        "num_classes": len(train_data.fam_to_idx),
        "pad_id": 0,
        "bos_id": 1,
        "eos_id": 2,

        # Model parameters
        "num_epochs": args.num_epochs,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,

        # MLP specific
        "context_size": 3, # 16, 32, 64, 128

        # LSTM specific
        "num_layers": args.num_layers,
        "dropout": args.dropout,

        "max_length": 128,
        "learning_rate": args.learning_rate
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if config["model_name"] == 'MLP':
        model = MLP.AutoregressiveMLP(
            vocab_size = config["vocab_size"], 
            context_size = config["context_size"], 
            embed_dim = config["embed_dim"], 
            hidden_dim = config["hidden_dim"]
        ).to(device)
        train = MLP.train
        validation = MLP.validation

    elif config["model_name"] == 'LSTM':
        model = LSTM.LSTMModel(
            vocab_size = config["vocab_size"],
            embed_dim = config["embed_dim"], 
            hidden_dim = config["hidden_dim"],
            num_layers = config["num_layers"],
            dropout = config["dropout"]
        ).to(device)
        train = LSTM.train
        validation = LSTM.validation

    elif config["model_name"] == 'ClassifierLSTM':
        model = ClassifierLSTM.NoisyPredictor(
            vocab_size = config["vocab_size"],
            embed_dim = config["embed_dim"],
            hidden_dim = config["hidden_dim"],
            num_layers = config["num_layers"],
            num_classes = config["num_classes"],
            dropout = config["dropout"]
        ).to(device)
        train = ClassifierLSTM.train
        validation = ClassifierLSTM.validation


    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "best_val_loss": "",
        "best_val_acc": "",
        "best_epoch": ""
    }

    best_val_loss = float("inf")
    patience = 10
    counter = 0

    # Training and validation loop
    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train(model, train_loader, optimizer, config, device)
        val_loss, val_acc = validation(model, val_loader, optimizer, config, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # find the best model
        if val_loss < best_val_loss:
            counter = 0

            history["best_val_loss"] = val_loss
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch + 1
            torch.save(model.state_dict(), f"src/protgen/models/best_{config["model_name"]}_model.pt")
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping")
            break

    with open(f"src/protgen/models/{config["model_name"]}_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    with open(f"src/protgen/models/{config["model_name"]}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print(f"The best model was saved")

if __name__ == "__main__":
    main()