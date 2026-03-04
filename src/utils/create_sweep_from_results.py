import pandas as pd
import yaml
import argparse
import numpy as np

def to_python_type(val):
    if isinstance(val, np.generic):
        return val.item()
    return val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV from W&B export")
    parser.add_argument("--output", default="rerun_sweep.yaml", help="Output sweep YAML")
    parser.add_argument("--project", default="fakenews-gnn")
    parser.add_argument("--script", default="train_crossval.py")
    parser.add_argument(
        "--include_cols", nargs="+",
        default=[
            "batch_size", "hidden_channels", "num_layers",
            "conv_type", "dropout", "use_batchnorm",
            "pooling", "optimizer", "lr", "weight_decay", "epochs"
        ],
        help="Explicit list of columns to include as hyperparameters"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Keep only included columns
    missing_cols = [c for c in args.include_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in CSV: {missing_cols}")
    df = df[args.include_cols]

    # Convert numpy types to python types
    for col in df.columns:
        df[col] = df[col].apply(to_python_type)

    sweep_yaml = {
        "program": args.script,
        "method": "grid",  # Only sweeping over config_id
        "metric": {"name": "val_loss", "goal": "minimize"},
        "command": [
            "${env}",
            "python",
            "${program}",
            "--project",
            args.project
        ],
        "parameters": {
            "config_id": {"values": list(range(len(df)))}
        }
    }

    # Save CSV with only included hyperparameters
    df.to_csv("sweep_configs.csv", index=False)
    with open(args.output, "w") as f:
        yaml.safe_dump(sweep_yaml, f, sort_keys=False)

    print(f"Sweep YAML saved to {args.output}")
    print(f"Configs saved to sweep_configs.csv")
    print(f"Total runs: {len(df)}")

if __name__ == "__main__":
    main()
