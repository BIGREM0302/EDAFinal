2025/06/22

-------------------------------------Pipeline Stage1 - Feature Extraction-------------------------------

##### Prepare Dataset (Can ignore this since data has already been prepared in correct directories)

g++ SVMparser.cpp -o SVMparser
./SVMparser

It will read all trojaned circuit in release_all and extract 5 parameters of each node and store it to ./training_data_for_svm

Furthermore, it also saves all graph information to ./training_data_for_svm/info/designxx

-------------------------------------Pipeline Stage2 - Phase 2 - SVM-------------------------------------

##### Virtual Environment

python3.11 -m venv venv

source ./venv/bin/activate

pip install -r requirements.txt

or

conda create -n GNN python=3.11

conda activate GNN

##### Train and Inference

-------------------------------------Pipeline Stage3 - Connectivity Filter-------------------------------

g++ ConnectUpdate.cpp -o ConnectUpdate
./ConnectUpdate --input a --g b --gs c

a: which design you want to filter now, 0-19
b:
1 -> for golden, it in turn won't filter since the answer will be directly read
// By setting --g 1 you can see the group structure of correct trojanned gate
0 -> activate filter function
// By setting --g 0 you can see the group structure of the output from SVM
c: the threshold of group size you want, 1 can have good performance

For example: ./ConnectUpdate --input 13 --g 0 --gs 1

The output file will be stored in ./filter

There will be information logged on terminal, which contains not only the group structure but also the score result before and after different filtering method(only maximum group / groups whose size > gs)

- The output file is filtered by the threshold method instead of by only choosing maximum group
