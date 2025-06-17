Utilization:
Can convert all design.v to the data type that suitable for GNN input

Which one to convert? modify line 107's ifstream data path

1.  g++ GNNparser.cpp -o GNNparser

./GNNparser

output file:
// for python's input
GNNedges.csv
GNNnodetypes.csv
// for debug
edges.csv
nodetypes.csv

2.  // python will read the generated csv and convert to the data type for GNN
    python3 -m venv venv
    source ./venv/bin/activate
    pip install -r requirements.txt

Data can be returned
