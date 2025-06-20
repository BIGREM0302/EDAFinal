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
    def __init__(self, root_dir, transform=None, pre_transform=None, allow_missing_label=False):
        self.allow_missing_label = allow_missing_label
        super().__init__(root_dir, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    @property
    def processed_file_names(self):
        return ['design_graphs.pt']

    def process(self):
        graphs = []
        for design_dir in sorted(glob.glob(os.path.join(self.raw_dir, '*'))):
            print(design_dir)
            print("a")
            # expected file structure:
            # root -
            #      design1/
            #              GNNedges.csv
            #              GNNnodetypes.csv
            #              label.txt
            epath = os.path.join(design_dir, 'GNNedges.csv')
            vpath = os.path.join(design_dir, 'GNNnodetypes.csv')
            lpath = os.path.join(design_dir, 'label.txt')

            edge_index = torch.tensor(
                pd.read_csv(epath).values.T,
                dtype=torch.long
            )

            x_df = pd.read_csv(vpath).drop(columns=['id', 'name'], errors='ignore')
            x = torch.tensor(x_df.values, dtype=torch.float)
            if self.allow_missing_label:
                # 原始 y 是一個整數，表示 Trojan 類型（1 ~ 10), 0 stands for no Trojan
                graphs.append(Data(x=x, edge_index=edge_index))
            else:
                y = int(open(lpath).read().strip())
                graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.long)))
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

        # 二元分類器，輸出 2 個 logits
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
        logits = self.head(pooled) # logics[0]: probability that it's trojaned/ logics[1]: probability that it's trojanned
        return logits
        # 回傳 class 1 的機率（accept 機率）
        #probs = F.softmax(logits, dim=1)
        #return probs[:, 1]  # 回傳每個 graph 是 Trojan 的機率


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
            # simply get the maximum one
            preds = model(batch).argmax(dim=1)
            correct += (preds == batch.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

if __name__ == '__main__':
    root_dir = './dataset'
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    epochs     = 1000
    lr         = 2e-3
    wd         = 1e-4

    # 1) 載入完整資料集（y 是 0~10）
    full_dataset = VerilogDataset(root_dir)
    print(len(full_dataset))
    # 為了 stratify，我们先收集原始多類別標籤
    all_labels = [full_dataset[i].y.item() for i in range(len(full_dataset))]
    print(all_labels)
    print(f'\n=== Training classifier ===')

    # 建立 indices 列表並用 stratify 分割
    _, idx_te = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.5,
        #stratify=all_labels,
        random_state=42
    )
    # our train data size too small
    idx_tr = list(range(len(full_dataset)))

    # 為了做「一 vs. 其餘」，我們需要動態修改每個 sample 的 label
    def make_binary_dataset(indices):
        graphs = []
        for idx in indices:
            data = full_dataset[idx]
            # 如果原始 label == trojan_type，則新的 y=1，否則 y=0
            y_bin = 1 if data.y.item() >= 1 else 0
            graphs.append(Data(x=data.x, edge_index=data.edge_index, y=torch.tensor([y_bin], dtype=torch.long)))
        return graphs

    train_graphs = make_binary_dataset(idx_tr)
    test_graphs  = make_binary_dataset(idx_te)

    loader_tr = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    loader_te = DataLoader(test_graphs,  batch_size=batch_size)

    model = TrojanDetector(in_dim=full_dataset.num_node_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(1, epochs+1):
        loss = train_epoch(model, loader_tr, optimizer, device)
        if epoch % 10 == 0 or epoch == epochs:
            acc = evaluate(model, loader_te, device)
            print(f'  Epoch {epoch:03d}  loss={loss:.4f}  val_acc={acc:.3f}')

    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)  # 如果資料夾已存在，不會報錯
    model_path = f'./models/trojan_detector.pt'
    torch.save(model.state_dict(), model_path)
    print(f'  >> Saved model to {model_path}')
