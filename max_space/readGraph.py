import torch
import pandas as pd
from torch_geometric.data import Data

# read edge list
df = pd.read_csv("GNNedges.csv")
edge_index = torch.tensor(df.values.T, dtype=torch.long)  # shape: [2, num_edges]
print(edge_index.shape)

# node features
node_features = pd.read_csv("GNNnodetypes.csv")
# remove the column id // if we have name column afterwards, remove as well
x = torch.tensor(node_features.drop(columns=["id","name"]).values, dtype=torch.float)
print(x.shape)

# construct Data object
data = Data(x=x, edge_index=edge_index)

print(data)