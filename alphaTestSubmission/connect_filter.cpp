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
#include <map>

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
vector<string> max_group_gates;
vector<string> threshold_group_gates;


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

double calculate_filtered_F1_score(vector<string> try_gates){
    int TP, FP, FN, TN;
    double precision, recall;
    TP = 0;
    FP = 0;
    FN = 0;
    TN = 0;
    for(const auto& a:try_gates){
        //estimate trojan
        if (find(golden_gates.begin(), golden_gates.end(), a) != golden_gates.end()) {
            TP += 1;
        } else {
            FP += 1;
        }
    }
    for(const auto& a:golden_gates){
        //estimate trojan
        if (find(try_gates.begin(), try_gates.end(), a) == try_gates.end()) {
            FN += 1;
        }
    }
    TN = nodes.size() - TP - FN - FP;
    precision = 1.0 * TP / (TP + FP);
    recall = 1.0 * TP / (TP + FN);
    cout << "======Score Result======" << endl;
    cout << "TP: " << TP << ", FP: " << FP << endl;
    cout << "FN: " << FN << ", TN: " << TN << endl;
    if(TP == 0) return 0;
    return (2 * (precision * recall))/(precision + recall);
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
    if (TP == 0) return 0;
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
    string golden = "0";
    int gs = 1;
    bool filter = true;

    string base_address = "./parser_result/info";
    string test_address;
    //string golden_address;
    test_address = "picked_gates.txt";
    
    load_nodes(base_address + "/nodes.txt");
    load_node_id_map(base_address + "/node_id_map.txt");
    load_wire_to_node(base_address + "/wire_to_node.txt");

    // load currently picked nodes (subset of all nodes)
    load_test_data(test_address);
    //load_golden_data(golden_address);
    //show_current_id_node_map();

    vector<vector<string>> groups;

    groups = connect_component_detection();
    
    cout << "=============groups============" << endl;
    int group_id = 0;
    vector<int> candidate_group_id;
    int max_group_id = 0;
    int max_group_size = 0;
    for(const auto& group: groups){
        group_id ++;
        cout << "Group_id = " << group_id << endl;
        cout << "group_size = " << group.size() << endl;
        
        if(group.size() > max_group_size){
            max_group_size = group.size();
            max_group_id = group_id;
        }
        
        if(group.size() > gs){
            candidate_group_id.push_back(group_id);
        }

        for(const auto& element: group){
            cout << element << ",";
        }
        cout << endl;
    }
    //double f1score;
    //f1score = calculate_F1_score();
    //cout << "F1 score =" << f1score << endl;
    
    if(filter){
        string filterfilepath = "filter_result.txt";
        ofstream filterfile(filterfilepath);
        cout << "=========Start Filtering========" << endl;
        cout << "Extract group size > " << gs << " will be extracted" << endl;
        if(candidate_group_id.size() <= 0) cout << "gs is too high! no group is extracted T__T" << endl;
        threshold_group_gates.clear();
        for(const auto& gid : candidate_group_id){
            if(gid < 1) cout << "Some bad occurs" << endl;
            threshold_group_gates.insert(threshold_group_gates.end(), groups[gid-1].begin(), groups[gid-1].end());
        }
        cout << "=========Write output file=========" << endl;
        for(const auto& newly_picked_gate: threshold_group_gates)
            filterfile << newly_picked_gate << endl;
        cout << "Write filtered gates to " << filterfilepath << endl;
    }
    
    return 0;
}