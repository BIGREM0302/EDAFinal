python3.11 -m venv venv
source ./venv/bin/activate

python -m pip install -U pip setuptools wheel
pip install torch==2.7.0 --extra-index-url https://download.pytorch.org/whl/cpu

TORCH=2.7.0
WHEEL_URL=https://data.pyg.org/whl/torch-${TORCH}+cpu.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f ${WHEEL_URL}

pip install torch_geometric==2.6.1

pip install -r ./GNN/requirements.txt
