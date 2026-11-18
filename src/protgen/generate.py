import argparse
import pandas as pd
import json
import torch

from protgen.models import MLP, LSTM

def parse_args():
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--model-name", type=str, default='LSTM')
    parser.add_argument("--output-number", type=int, default=200)
    return parser.parse_args()

def main():
    args = parse_args()
    with open(f"src/protgen/models/{args.model_name}_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if config["model_name"] == 'MLP':
        model = MLP.AutoregressiveMLP(
            vocab_size = config["vocab_size"], 
            context_size = config["context_size"], 
            embed_dim = config["embed_dim"], 
            hidden_dim = config["hidden_dim"]
        ).to(device)
        generate = MLP.generate

    elif config["model_name"] == 'LSTM':
        model = LSTM.LSTMModel(
            vocab_size = config["vocab_size"],
            embed_dim = config["embed_dim"], 
            hidden_dim = config["hidden_dim"],
            num_layers = config["num_layers"],
            dropout = config["dropout"]
        ).to(device)
        generate = LSTM.generate

    # Generate sequences
    vocab = {v: k for k, v in config["vocab"].items()}
    model.load_state_dict(
        torch.load(f"src/protgen/models/best_{config["model_name"]}_model.pt", map_location=device, weights_only=True)
    )
    rows = []

    for num in range(args.output_number):
        generated = generate(model, config, device, config["max_length"])
        sequence = ''.join(
            vocab[i] for i in generated
        )
        rows.append({'sequence': sequence})
        
    pd.DataFrame(rows).to_csv(f"results/sequences/generated_{config["model_name"]}_{args.output_number}.csv", index=False)


if __name__ == "__main__":
    main()