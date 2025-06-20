import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from gnn_train import TrojanDetector, VerilogDataset

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

def infer_batch(graphs, model, device, threshold=0.5, batch_size=8):
    loader = DataLoader(graphs, batch_size=batch_size)
    results = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)[:, 1]  # class 1 是 Trojan
            for p in probs:
                prob = p.item()
                pred = prob >= threshold
                results.append((pred, prob))
    return results

if __name__ == "__main__":
    root_dir    = "./testset"
    model_path  = "./models/trojan_detector_type1.pt"  # ✅ 只載入這個模型
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold   = 0.6
    batch_size  = 4

    dataset = VerilogDataset(root_dir, allow_missing_label=True)
    graphs  = [dataset[i] for i in range(len(dataset))]

    model = load_model(model_path, in_dim=dataset.num_node_features, device=device)

    results = infer_batch(graphs, model, device, threshold, batch_size)

    for idx, (pred, prob) in enumerate(results):
        if pred:
            print(f"樣本 {idx:3d}: 預測 **有** Trojan，機率 = {prob:.3f}")
        else:
            print(f"樣本 {idx:3d}: 預測 **無** Trojan，機率 = {prob:.3f}")
