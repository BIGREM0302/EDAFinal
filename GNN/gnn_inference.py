import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from your_module import TrojanDetector

def load_models(model_dir, in_dim, device):
    """
    載入 10 個 classifier，返回 list of models。
    model_dir: 模型檔案放置的資料夾路徑
    in_dim:   node feature 維度
    device:   'cpu' 或 'cuda'
    """
    models = []
    for t in range(10):
        path = f"{model_dir}/trojan_detector_type{t}.pt"
        model = TrojanDetector(in_dim=in_dim).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models

def infer_one(graph: Data, models, device, threshold=0.5):
    """
    對單一 graph 做推論：
      回傳 (predicted_type, scores) 
      若都未超過 threshold，predicted_type 回傳 None
    """
    # 確保 graph 包含 batch 屬性
    if not hasattr(graph, 'batch'):
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    graph = graph.to(device)
    
    scores = []
    with torch.no_grad():
        for model in models:
            logits = model(graph)              # [1, 2]
            prob = F.softmax(logits, dim=1)[0,1].item()  # 取 label=1 的機率
            scores.append(prob)
    
    # 找出超過 threshold 的最大機率
    valid = [(i, p) for i, p in enumerate(scores) if p >= threshold]
    if not valid:
        return None, scores
    # 取機率最高的那個
    pred_type, best_p = max(valid, key=lambda x: x[1])
    return pred_type, scores

def infer_batch(graphs, models, device, threshold=0.5, batch_size=8):
    """
    對多個 graph 做推論，回傳 list of (predicted_type, scores)。
    """
    loader = DataLoader(graphs, batch_size=batch_size)
    results = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            # 對每個模型算一個 [batch_size,] 的 vector
            all_probs = []
            for model in models:
                logits = model(batch)
                probs = F.softmax(logits, dim=1)[:,1]   # 取 label=1 的機率
                all_probs.append(probs)                # list of tensors
            # shape: [10, batch_size] -> tensor [batch_size, 10]
            probs_mat = torch.stack(all_probs, dim=0).t()
            for p_vec in probs_mat:
                # p_vec 是長度 10 的機率向量
                valid = [(i, p.item()) for i,p in enumerate(p_vec) if p.item() >= threshold]
                if not valid:
                    results.append((None, p_vec.tolist()))
                else:
                    pred, _ = max(valid, key=lambda x: x[1])
                    results.append((pred, p_vec.tolist()))
    return results

if __name__ == "__main__":
    import os
    from your_module import VerilogDataset  # 改成你實際的 dataset 定義

    # 參數設定
    root_dir    = "./dataset"
    model_dir   = "./"            # trojan_detector_type0.pt ... 檔案所在資料夾
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold   = 0.6             # 可依驗證結果調整門檻
    batch_size  = 4

    # 1) 載入 dataset，並挑選要推論的樣本 (這裡示範全資料集)
    dataset = VerilogDataset(root_dir)
    graphs  = [dataset[i] for i in range(len(dataset))]  # 或是你要推論的 subset

    # 2) 載入所有模型
    models = load_models(model_dir, in_dim=dataset.num_node_features, device=device)

    # 3) 進行 batch 推論
    results = infer_batch(graphs, models, device, threshold, batch_size)

    # 4) 顯示結果
    for idx, (pred, scores) in enumerate(results):
        if pred is None:
            print(f"樣本 {idx:3d}: 無 Trojan，最高機率 {max(scores):.3f}")
        else:
            print(f"樣本 {idx:3d}: 預測 Trojan 類型 {pred}，機率 {scores[pred]:.3f}")
