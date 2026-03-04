import wandb
import os
import sys
import torch
import torch.nn.functional as F
import argparse
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Subset
from torch_geometric.utils import dropout_edge
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.models.Fakenews_GNN import Fakenews_GNN
from src.utils.data_loader import FNNDataset, add_bot_score_feature


# -------------------------------------------------------
# Masking / Dropout functions
# -------------------------------------------------------

def mask_features(x, mask_prob):
    if mask_prob <= 0:
        return x
    mask = torch.rand_like(x) < mask_prob
    return x * (~mask)


def apply_edge_dropout(edge_index, drop_prob):
    if drop_prob <= 0:
        return edge_index
    edge_index, _ = dropout_edge(edge_index, p=drop_prob)
    return edge_index


# -------------------------------------------------------
# Setup
# -------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--project", default="fakenews-gnn", type=str)
args = parser.parse_args()

wandb.init(project=args.project)
config = wandb.config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Fakenews_GNN(
    in_channels=11,
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
gos_map_path = data_dir / "gos_id_twitter_mapping.pkl"
root = os.path.join(root_dir, "data")

dataset_gos = FNNDataset(root=root, name="gossipcop", feature="profile")
add_bot_score_feature(dataset_gos, gos_map_path, "./scores_gos.json")

pol_map_path = data_dir / "pol_id_twitter_mapping.pkl"
dataset_pol = FNNDataset(root=root, name="politifact", feature="profile")
add_bot_score_feature(dataset_pol, pol_map_path, "./scores_pol.json")

train_dataset_gos = Subset(dataset_gos, dataset_gos.test_idx)
val_dataset_gos   = Subset(dataset_gos, dataset_gos.train_idx)
test_dataset_gos  = Subset(dataset_gos, dataset_gos.val_idx)

train_dataset_pol = Subset(dataset_pol, dataset_pol.test_idx)
val_dataset_pol   = Subset(dataset_pol, dataset_pol.train_idx)
test_dataset_pol  = Subset(dataset_pol, dataset_pol.val_idx)

train_dataset = ConcatDataset([train_dataset_gos, train_dataset_pol])
len_gos = len(train_dataset_gos)
len_pol = len(train_dataset_pol)
weights_gos = torch.ones(len_gos) / len_gos
weights_pol = torch.ones(len_pol) / len_pol
sample_weights = torch.cat([weights_gos, weights_pol])
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = PyGDataLoader(
    train_dataset,
    batch_size=config.batch_size,
    sampler=sampler
)

val_loader_gos = PyGDataLoader(val_dataset_gos, batch_size=config.batch_size, shuffle=False)
val_loader_pol = PyGDataLoader(val_dataset_pol, batch_size=config.batch_size, shuffle=False)

test_loader_gos = PyGDataLoader(test_dataset_gos, batch_size=config.batch_size, shuffle=False)
test_loader_pol = PyGDataLoader(test_dataset_pol, batch_size=config.batch_size, shuffle=False)


# -------------------------------------------------------
# Validation / Testing function
# -------------------------------------------------------

def evaluate(loader):
    correct, total, loss_sum = 0, 0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss_sum += F.cross_entropy(out, data.y).item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return loss_sum / len(loader), correct / total


# -------------------------------------------------------
# Training loop (with masking + edge dropout)
# -------------------------------------------------------

feature_mask_prob = getattr(config, "feature_mask_prob", 0.05)
edge_drop_prob = getattr(config, "edge_drop_prob", 0.1)

for epoch in range(config.epochs):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)

        # --------------------------------------
        # Apply masking augmentations
        # --------------------------------------
        data.x = mask_features(data.x, mask_prob=feature_mask_prob)
        data.edge_index = apply_edge_dropout(data.edge_index, drop_prob=edge_drop_prob)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # --- Validation on both datasets ---
    val_loss_gos, val_acc_gos = evaluate(val_loader_gos)
    val_loss_pol, val_acc_pol = evaluate(val_loader_pol)

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss_gos": val_loss_gos,
        "val_acc_gos": val_acc_gos,
        "val_loss_pol": val_loss_pol,
        "val_acc_pol": val_acc_pol
    })

    print(
        f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, "
        f"Val GossipCop Acc: {val_acc_gos:.4f}, Val Politifact Acc: {val_acc_pol:.4f}"
    )


# -------------------------------------------------------
# Final Tests
# -------------------------------------------------------

test_loss_gos, test_acc_gos = evaluate(test_loader_gos)
test_loss_pol, test_acc_pol = evaluate(test_loader_pol)

wandb.log({
    "test_loss_gos": test_loss_gos,
    "test_acc_gos": test_acc_gos,
    "test_loss_pol": test_loss_pol,
    "test_acc_pol": test_acc_pol
})

print(f"Test GossipCop: Loss {test_loss_gos:.4f}, Acc {test_acc_gos:.4f}")
print(f"Test Politifact: Loss {test_loss_pol:.4f}, Acc {test_acc_pol:.4f}")
