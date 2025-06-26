# 基礎映像（含 GPU 支援）
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 更新 + 安裝 Python + 系統工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件（根據你給的 import）
RUN pip3 install --no-cache-dir \
    torch==2.7.1+cu126 \
    torchvision==0.15.2+cu126 \
    torchaudio==2.0.2+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

RUN pip3 install --no-cache-dir pandas joblib numpy scikit-learn torch_geometric

# 建立工作資料夾
WORKDIR /app

# 複製所有檔案
COPY connect_filter /app/connect_filter
COPY parser /app/parser
COPY gnn_inference.py /app/gnn_inference.py
COPY svm_infer.py /app/svm_infer.py
COPY models/trojan_detector.pt /app/models/trojan_detector.pt
COPY models/trojan_svm.joblib /app/models/trojan_svm.joblib

# 指定主程式為入口
COPY cada1034_alpha /app/cada1034_alpha
RUN chmod +x /app/cada1034_alpha
