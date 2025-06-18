import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

class VerilogDataset(InMemoryDataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        super().__init__(root_dir, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['design_graphs.pt']

    def process(self):
        graphs = []
        for design_dir in sorted(glob.glob(os.path.join(self.root, '*'))):
            epath = os.path.join(design_dir, 'GNNedges.csv')
            vpath = os.path.join(design_dir, 'GNNnodetypes.csv')
            lpath = os.path.join(design_dir, 'label.txt')

            edge_index = torch.tensor(
                pd.read_csv(epath).values.T,
                dtype=torch.long
            )

            x_df = pd.read_csv(vpath).drop(columns=['id', 'name'], errors='ignore')
            x = torch.tensor(x_df.values, dtype=torch.float)

            y = torch.tensor([int(open(lpath).read().strip())], dtype=torch.long)

            graphs.append(Data(x=x, edge_index=edge_index, y=y))

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

class TrojanDetector(nn.Module):
    def __init__(self, in_dim, hidden=64, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)

        def mlp():
            return nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )

        self.convs = nn.ModuleList([GINConv(mlp()) for _ in range(num_layers)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)])

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x).relu()
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x).relu()
        pooled = global_mean_pool(x, batch)
        return self.head(pooled)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch).argmax(dim=1)
            correct += (preds == batch.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

if __name__ == '__main__':
    root_dir = './dataset'
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    epochs     = 100
    lr         = 2e-3
    wd         = 1e-4
    model_save_path = 'trojan_detector.pt'

    # 1) load dataset
    dataset = VerilogDataset(root_dir)
    idx_tr, idx_te = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        stratify=[dataset[i].y.item() for i in range(len(dataset))]
    )
    loader_tr = DataLoader(dataset[idx_tr], batch_size=batch_size, shuffle=True)
    loader_te = DataLoader(dataset[idx_te], batch_size=batch_size)

    # 2) init model & optimizer
    model = TrojanDetector(in_dim=dataset.num_node_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 3) training loop
    for epoch in range(1, epochs+1):
        loss = train_epoch(model, loader_tr, optimizer, device)
        if epoch % 10 == 0:
            acc = evaluate(model, loader_te, device)
            print(f'Epoch {epoch:03d}  loss={loss:.4f}  val_acc={acc:.3f}')

    # 4) save model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
