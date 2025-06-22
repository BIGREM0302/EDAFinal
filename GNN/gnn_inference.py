import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from gnn_train import TrojanDetector

import os
from gnn_train import VerilogDataset

def load_models(model_dir, in_dim, device):
    models = []
    for t in range(1,11):
        path = f"{model_dir}/trojan_detector_type{t}.pt"
        model = TrojanDetector(in_dim=in_dim).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models

def infer_one(graph: Data, models, device, threshold=0.5):
    if not hasattr(graph, 'batch'):
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    graph = graph.to(device)

    scores = []
    with torch.no_grad():
        for model in models:
            logits = model(graph)
            prob = F.softmax(logits, dim=1)[0,1].item()
            scores.append(prob)

    valid = [(i, p) for i, p in enumerate(scores) if p >= threshold]
    if not valid:
        return None, scores

    pred_type, best_p = max(valid, key=lambda x: x[1])
    return pred_type, scores

def infer_batch(graphs, models, device, threshold=0.5, batch_size=8):
    loader = DataLoader(graphs, batch_size=batch_size)
    results = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            all_probs = []
            for model in models:
                logits = model(batch)
                probs = F.softmax(logits, dim=1)[:,1]
                all_probs.append(probs)
            probs_mat = torch.stack(all_probs, dim=0).t()
            for p_vec in probs_mat:
                valid = [(i, p.item()) for i,p in enumerate(p_vec) if p.item() >= threshold]
                if not valid:
                    results.append((None, p_vec.tolist()))
                else:
                    pred, _ = max(valid, key=lambda x: x[1])
                    results.append((pred, p_vec.tolist()))
    return results

if __name__ == "__main__":

    root_dir    = "./testset"
    model_dir   = "./models"
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold   = 0.6
    batch_size  = 4

    dataset = VerilogDataset(root_dir, allow_missing_label=True)
    graphs  = [dataset[i] for i in range(len(dataset))]

    models = load_models(model_dir, in_dim=dataset.num_node_features, device=device)

    results = infer_batch(graphs, models, device, threshold, batch_size)

    count = 0
    correct_count = 0
    for idx, (pred, scores) in enumerate(results):
        count = count + 1
        if(idx < 20):
            if pred: 
                print("correct!")
                correct_count = correct_count+1
            else:
                print("wrong...")
        else:
            if pred:
                print("wrong...")
            else:
                correct_count = correct_count+1
                print("correct!")
        if pred is None:
            print(f"樣本 {idx:3d}: 無 Trojan，最高機率 {max(scores):.3f}")
        else:
            print(f"樣本 {idx:3d}: 預測 Trojan 類型 {pred}，機率 {scores[pred]:.3f}")
    print(f"正確比例: {correct_count} / {count}")
