import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import os
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GINConv, global_mean_pool

def load_graph(graph_dir: str) -> Data:
    edge_path = os.path.join(graph_dir, "GNNedges.csv")
    node_path = os.path.join(graph_dir, "GNNnodetypes.csv")

    edge_index = torch.tensor(pd.read_csv(edge_path).values.T, dtype=torch.long)

    node_df = pd.read_csv(node_path).drop(columns=['id', 'name'], errors='ignore')
    x = torch.tensor(node_df.values, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)

class TrojanDetector(nn.Module):
    def __init__(self, in_dim, hidden=128, num_layers=4):
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
            x = F.dropout(x, p=0.3, training=self.training)
        pooled = global_mean_pool(x, batch)
        logits = self.head(pooled) # logics[0]: probability that it's trojaned/ logics[1]: probability that it's trojanned
        return logits
        # 回傳 class 1 的機率（accept 機率）
        #probs = F.softmax(logits, dim=1)
        #return probs[:, 1]  # 回傳每個 graph 是 Trojan 的機率

def load_model(model_path, in_dim, device):
    model = TrojanDetector(in_dim=in_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def infer_one(graph: Data, model, device, threshold=0.5):
    if not hasattr(graph, 'batch'):
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    graph = graph.to(device)

    with torch.no_grad():
        logits = model(graph)
        prob = F.softmax(logits, dim=1)[0, 1].item()  # 機率為 Trojan
        pred = prob >= threshold

    return pred, prob

if __name__ == "__main__":
    root_dir    = "./parser_result"
    model_path  = "./models/trojan_detector.pt"  # ✅ 只載入這個模型
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold   = 0.6
    batch_size  = 4
    output = "hasTrojan.txt"

    graph = load_graph(root_dir)
    in_dim = graph.num_node_features
    # 載入模型
    model = load_model(model_path, in_dim=in_dim, device=device)

    # 推論
    pred, prob = infer_one(graph, model, device, threshold)

    # 寫入結果
    with open(output, "w") as f:
        f.write(f"{pred}\n")  # 1 或 0

    print(f"✅ 推論完成：{'TROJANED (1)' if pred else 'NO_TROJAN (0)'}, 機率={prob:.3f}")
    print(f"✍️ 結果已寫入：{output}")
