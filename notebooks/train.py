import wandb
import os
import sys
import torch
import torch.nn.functional as F
import argparse
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from src.models.Fakenews_GNN import Fakenews_GNN
from src.utils.data_loader import FNNDataset, add_bot_score_feature


parser = argparse.ArgumentParser()
parser.add_argument("--project", default="fakenews-gnn", type=str)
args = parser.parse_args()

wandb.init(project=args.project)
config = wandb.config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Fakenews_GNN(
    in_channels=301,
    hidden_channels=config.hidden_channels,
    out_channels=2,
    num_layers=config.num_layers,
    conv_type=config.conv_type,
    dropout=config.dropout,
    use_batchnorm=config.use_batchnorm,
    pooling=config.pooling
).to(device)

optimizer_cls = torch.optim.Adam if config.optimizer == "Adam" else torch.optim.AdamW
optimizer = optimizer_cls(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

data_dir = Path("../data/raw/")
pol_map_path = data_dir / "gos_id_twitter_mapping.pkl"
root = os.path.join(root_dir, "data")
dataset = FNNDataset(root=root, name="gossipcop", feature="spacy")
print(f"Loaded Regular Dataset")
add_bot_score_feature(dataset, pol_map_path, "./scores_gos.json")

train_dataset = Subset(dataset, dataset.test_idx)
val_dataset   = Subset(dataset, dataset.train_idx)
test_dataset  = Subset(dataset, dataset.val_idx)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# Training loop
for epoch in range(config.epochs):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            val_loss += F.cross_entropy(out, data.y).item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    val_loss /= len(val_loader)
    val_acc = correct / total

    # Logging
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Test evaluation
model.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        test_loss += F.cross_entropy(out, data.y).item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
test_loss /= len(test_loader)
test_acc = correct / total

wandb.log({"test_loss": test_loss, "test_acc": test_acc})
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")