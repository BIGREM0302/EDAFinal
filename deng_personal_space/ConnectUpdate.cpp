#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>
#include <climits>
#include <queue>
#include <unordered_set>

using namespace std;

struct Node {
    string name;
    string type;
    string output;
    vector<string> inputs;
};

unordered_map<string, Node> nodes;                   // node_name -> Node
unordered_map<string, string> wire_to_node;          // wire_name -> node_name
unordered_map<string, int> node_id_map;              // node_name -> id
unordered_map<int, string> id_node_map;             // id -> node_name

vector<string> picked_gates;
vector<string> golden_gates;


string pad2(int i) {
    return (i < 10 ? "0" : "") + to_string(i);
}

class UnionFind {
public:
    unordered_map<string, string> parent;

    void make_set(string x) {
        parent[x] = x;
    }

    string find(string x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(string x, string y) {
        string rx = find(x);
        string ry = find(y);
        if (rx != ry)
            parent[ry] = rx;
    }

    void show(){
        for(const auto& [a, a_parent]:parent){
            cout << a << "'s parent is " << a_parent << endl;
        }
    }
};


void load_nodes(const string& filename) {
    nodes.clear();
    ifstream fin(filename);
    if (!fin) {
        cerr << "Failed to open file for reading: " << filename << endl;
        return;
    }

    string line;
    while (getline(fin, line)) {
        stringstream ss(line);
        string key, name, type, output;
        getline(ss, key, ',');
        getline(ss, name, ',');
        getline(ss, type, ',');
        getline(ss, output, ',');

        vector<string> inputs;
        string input;
        while (getline(ss, input, ',')) {
            inputs.push_back(input);
        }

        nodes[key] = Node{name, type, output, inputs};
    }
}

void load_node_id_map(const string& filename){
    node_id_map.clear();
    id_node_map.clear();
    ifstream fin(filename);
    string node_name;
    int node_id;
    while(fin >> node_name >> node_id){
        node_id_map[node_name] = node_id;
        id_node_map[node_id] = node_name;
    }
    fin.close();
}

void load_wire_to_node(const string& filename){
    wire_to_node.clear();
    ifstream fin(filename);
    string wire_name;
    string gate_name;
    while(fin >> wire_name >> gate_name){
        wire_to_node[wire_name] = gate_name;
    }
    fin.close();
}

void load_test_data(const string& filename){
    picked_gates.clear();
    ifstream fin(filename);
    string line;
    while(getline(fin, line)){
        if(nodes.count(line) > 0){
            picked_gates.push_back(line);
            //cout << line << "is initially chosen as trojaned gate" << endl;
        }
    }
    fin.close();
}

void load_golden_data(const string& filename){
    golden_gates.clear();
    ifstream fin(filename);
    string line;
    while(getline(fin, line)){
        if(nodes.count(line) > 0){
            golden_gates.push_back(line);
            //cout << line << "is initially chosen as trojaned gate" << endl;
        }
    }
    fin.close();
}

void show_current_id_node_map(){
    cout << "id->name mapping:" << endl;
    for(const auto& [id, name]: id_node_map){
        cout << id << "->" << name << endl;
    }
}

double calculate_F1_score(){
    int TP, FP, FN, TN;
    double precision, recall;
    TP = 0;
    FP = 0;
    FN = 0;
    TN = 0;
    for(const auto& a:picked_gates){
        //estimate trojan
        if (find(golden_gates.begin(), golden_gates.end(), a) != golden_gates.end()) {
            TP += 1;
        } else {
            FP += 1;
        }
    }
    for(const auto& a:golden_gates){
        //estimate trojan
        if (find(picked_gates.begin(), picked_gates.end(), a) == picked_gates.end()) {
            FN += 1;
        }
    }
    TN = nodes.size() - TP - FN - FP;
    precision = 1.0 * TP / (TP + FP);
    recall = 1.0 * TP / (TP + FN);
    cout << "======Score Result======" << endl;
    cout << "TP: " << TP << ", FP: " << FP << endl;
    cout << "FN: " << FN << ", TN: " << TN << endl;
    
    return (2 * (precision * recall))/(precision + recall);
}

vector<vector<string>> connect_component_detection(){
// picked gates
    unordered_set<string> subset(picked_gates.begin(), picked_gates.end());
    UnionFind uf;
    for (string node : subset) {
        //cout << "Create node in uf: " << node << endl;
        uf.make_set(node);
    }

    for (const auto& [u, node_u]: nodes) {
        for (string v : node_u.inputs) {
            // if u and wire_to_node[v] both in subset
            //cout << v << " is " << u << "'s input" << endl;
            if (subset.count(u) && subset.count(wire_to_node[v])) {
                //cout << wire_to_node[v] << " is " << u << "'s input" << endl;
                uf.unite(u, wire_to_node[v]);
                uf.unite(wire_to_node[v], u);
            }
        }
    }

    map<string, vector<string>> groups;
    for (string node : subset) {
        string root = uf.find(node);
        groups[root].push_back(node);
    }

    // print out uf
    //uf.show();

    vector<vector<string>> result;
    for (auto& [_, group] : groups)
        result.push_back(group);

    return result;
}

int main(int argc, char* argv[]){

    // should modify for more general case
    string j;
    string golden = "0";
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "--input" && i + 1 < argc) {
            j = argv[i+1];
            cout << j << endl;
            i++;
        } else if (arg == "--g" && i + 1 < argc) {
            golden = argv[i+1];
            cout << golden << endl;
        }
    }

    string base_address = "./training_data_for_svm/info/";
    string test_address;
    string golden_address;
    if(golden == "0"){
        test_address = "./Supervised_SVM/result/GNNfeature" + j + "_SVM.csv";
    }
    else test_address = "../release_all/trojan/result" + j + ".txt";
    golden_address = "../release_all/trojan/result" + j + ".txt";

    load_nodes(base_address + "design" + pad2(stoi(j)) + "/nodes.txt");
    load_node_id_map(base_address + "design" + pad2(stoi(j)) + "/node_id_map.txt");
    load_wire_to_node(base_address + "design" + pad2(stoi(j)) + "/wire_to_node.txt");

    // load currently picked nodes (subset of all nodes)
    load_test_data(test_address);
    load_golden_data(golden_address);
    //show_current_id_node_map();

    vector<vector<string>> groups;

    groups = connect_component_detection();
    
    cout << "=============groups============" << endl;
    int group_id = 0;
    for(const auto& group: groups){
        group_id ++;
        cout << "Group_id = " << group_id << endl;
        cout << "group_size = " << group.size() << endl;
        for(const auto& element: group){
            cout << element << ",";
        }
        cout << endl;
    }
    double f1score;
    f1score = calculate_F1_score();
    cout << "F1 score =" << f1score << endl;
    return 0;
}