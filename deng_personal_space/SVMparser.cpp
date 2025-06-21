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
namespace fs = std::filesystem;

// type to one hot encoding
unordered_map<string, int> type_to_index;
int num_types;
int num_features;
/*
for (const auto& [name, g] : nodes) {
    if (type_to_index.count(g.type) == 0) {
        type_to_index[g.type] = type_idx++;
    }
}
*/

bool valid(const string& test){
    return (test != "1'b1" && test != "1'b0");
}

// To expand array like ports (only input or output need)
vector<string> expand_ports(const string& line) {
    vector<string> result;

    smatch match;
    regex bit_range_regex(R"(\[(\d+):(\d+)\])");

    string range_part;
    string var_part;

    if (regex_search(line, match, bit_range_regex)) {
        int high = stoi(match[1]);
        int low = stoi(match[2]);
        range_part = match.str();
        var_part = line.substr(match.position() + match.length());
        var_part = regex_replace(var_part, regex(R"(;|\s)"), ""); // remove ; and space

        // aplit variable
        stringstream ss(var_part);
        string var;
        while (getline(ss, var, ',')) {
            for (int i = low; i <= high; ++i) {
                result.push_back(var + "[" + to_string(i) + "]");
            }
        }
    } else {
        // if no range
        string cleaned = regex_replace(line, regex(R"(;|\s)"), "");
        stringstream ss(cleaned);
        string var;
        while (getline(ss, var, ',')) {
            result.push_back(var);
        }
    }
    return result;
}

vector<string> extract_connections(const string& line) {
    //cout << line << endl;
    vector<string> results;
    regex pattern("\\.\\w+\\(([^()]+)\\)");
    auto begin = sregex_iterator(line.begin(), line.end(), pattern);
    auto end = sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        string connection = (*it)[1].str();  // 取 connected_signal
        //cout << connection << endl;
        results.push_back(connection);
    }

    return results;
}

struct Node {
    string name;
    string type;
    string output;
    vector<string> inputs;
};

// global maps and vectors
unordered_map<string, int> FFI;
unordered_map<string, int> FFO;
//unordered_map<string, int> PI;
vector<int> PI;
//unordered_map<string, int> PO;
vector<int> PO;
vector<bool> Trojaned;

unordered_map<string, int> LGFI;
unordered_map<string, Node> nodes;                   // node_name -> Node
unordered_map<string, string> wire_to_node;          // wire_name -> node_name
unordered_map<string, int> node_id_map;              // node_name -> id
unordered_map<int, string> id_node_map;             // id -> node_name
unordered_map<string, string> dff_dq_map;           // d <-> q
unordered_map<string, vector<string>> output_data_dependency;
// output_data_dependency[node_name] -> the vector that it connect to

vector<string> primary_input_node_name;
vector<string> primary_output_node_name;
vector<string> dff_node_name;
vector<string> dff_input_name;

vector<pair<int, string>> sorted_nodes; // (id, name)

// save nodes

void save_graph_info(const string& dirname){
    ofstream fout1(dirname + "wire_to_node.txt");
    ofstream fout2(dirname + "node_id_map.txt");
    for (const auto& [key, val] : wire_to_node) {
        fout1 << key << " " << val << "\n";
    }
    fout1.close();
    for (const auto& [key, val] : node_id_map) {
        fout2 << key << " " << val << "\n";
    }
    fout2.close();
}

void save_nodes(const string& dirname) {
    ofstream fout(dirname + "nodes.txt");
    if (!fout) {
        cerr << "Failed to open file for writing: " << (dirname+"nodes.txt") << endl;
        return;
    }

    for (const auto& [key, node] : nodes) {
        fout << key << "," << node.name << "," << node.type << "," << node.output;
        for (const auto& input : node.inputs) {
            fout << "," << input;
        }
        fout << "\n";
    }
    fout.close();
}

// 

void read_trojanned_label(ifstream& resultfile){
    string line;
    int line_count = 0;
    Trojaned.clear();
    for(int i = 0; i < nodes.size(); i++){
        Trojaned.push_back(false);
    }
    while(getline(resultfile, line)){
        line_count ++;
        cout << line_count << endl;
        if(nodes.count(line) > 0){
            Trojaned[node_id_map[line]] = true;   
            cout << line << "is trojaned gate" << endl;
        }
    }
}

int lgfi(string index){
    if(nodes[index].inputs.size() <= 0){
        return 0;
    }
    else {
        int temp = 0;
        for(const auto& ancestor:nodes[index].inputs){
            //cout << "Pair: " << index << " " << wire_to_node[ancestor] << endl;
            //cout << ancestor << ":" << nodes[ancestor].inputs.size() << endl;
            if(valid(ancestor))
            temp += (nodes[wire_to_node[ancestor]].inputs.size()<=0)?1:nodes[wire_to_node[ancestor]].inputs.size();
        }
        return temp;
    }
}

void show_current_id_node_map(){
    cout << "id->name mapping:" << endl;
    for(const auto& [id, name]: sorted_nodes){
        cout << id << "->" << name << endl;
    }
}

void update_if_smaller(vector<int>& old_dist, const vector<int>& new_dist) {
    for (int i = 0; i < old_dist.size(); ++i) {
        if (new_dist[i] == -1) continue; // -1 表示 new_dist[i] 不可達，跳過
        if (old_dist[i] == -1 || new_dist[i] < old_dist[i]) {
            old_dist[i] = new_dist[i];
        }
    }
}

void process_FFO(){
    for(const auto& [name, ffo_value] : FFO){
        if(ffo_value >= (INT_MAX-1)) FFO[name] = -1;
    }
}

void process_FFI(){
    for(const auto& [name, ffi_value] : FFI){
        if(ffi_value >= (INT_MAX-1)) FFI[name] = -1;
        else if(FFI[name] == 0) cout << "Something fucked" << endl;
        else FFI[name] = FFI[name]-1;
    }
}

void initvector(vector<int> &v, int n){
    for(int i = 0; i < n; i++){
        v.push_back(-1);
    }
}

void showvector(const vector<int> v){
    cout << "[";
    for(const auto& element : v){
        cout << element << ",";
    }
    cout << "]" << endl;
}

vector<int> pi(string start){
    // not use recursive method
    // start from output nodes
    int n = nodes.size(); // total # of nodes, use id to store
    vector<int> dist (n, -1);  // -1 表示還沒到過 // 
    queue<int> q;

    dist[node_id_map[start]] = 0;
    q.push(node_id_map[start]);

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        // from output go back, so use inputs
        for (const auto& offspring : output_data_dependency[id_node_map[node]]) {
            // wire name -> node name -> id
            int neighbor = node_id_map[offspring];
            if (dist[neighbor] == -1) {  // 沒拜訪過
                dist[neighbor] = dist[node] + 1;
                q.push(neighbor);
            }
        }
    }
    return dist;
}

vector<int> po(string start){
    // not use recursive method
    // start from output nodes
    int n = nodes.size(); // total # of nodes, use id to store
    vector<int> dist (n, -1);  // -1 表示還沒到過 // 
    queue<int> q;

    dist[node_id_map[start]] = 0;
    q.push(node_id_map[start]);

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        // from output go back, so use inputs
        for (const auto& ancestor : nodes[id_node_map[node]].inputs) {
            if(valid(ancestor)){
                // wire name -> node name -> id
                int neighbor = node_id_map[wire_to_node[ancestor]];
                if (dist[neighbor] == -1) {  // 沒拜訪過
                    dist[neighbor] = dist[node] + 1;
                    q.push(neighbor);
                }
            }
        }
    }
    return dist;
}

int ffi(string index){
    //cout << "enter" << index << endl;
    if(FFI.count(index) > 0){
        //cout << "CaseD" << endl;
        return FFI[index];
    }
    else if(nodes[index].output.empty()){
        //cout << "CaseE" << endl;
        FFI[index] = INT_MAX-1;
        return FFI[index];
    }
    else {
        if(output_data_dependency[index].size() <= 0){
            //cout << "CaseA" << endl;
            FFI[index] = INT_MAX-1;
            return FFI[index];
        }
        else{
            int temp = INT_MAX-1;
            for(const auto& offspring : output_data_dependency[index]){
                //cout << "CaseB" << endl;
                if(nodes[offspring].type=="dff"){
                    temp = 0;
                    break;
                }
                if(ffi(offspring) < temp){
                    temp = FFI[offspring];
                }
            }
            //cout << "CaseC" << endl;
            FFI[index] = (temp >= (INT_MAX-1))? (INT_MAX-1) : (temp+1);
            return FFI[index];
        }
    }
}

int ffo(string index){
    if(FFO.count(index) > 0){
        return FFO[index];
    }
    else if(nodes[index].type == "dff"){
        // itself is a dff output
        FFO[index] = 0;
        return FFO[index];
    }
    else {
        if(nodes[index].inputs.size() <= 0){
            FFO[index] = INT_MAX-1;
            return FFO[index];
        }
        else{
            int temp = INT_MAX-1;
            for(const auto& ancestor : nodes[index].inputs){
                if(valid(ancestor))
                if(ffo(wire_to_node[ancestor]) < temp) temp = FFO[wire_to_node[ancestor]];
            }
            FFO[index] = (temp >= (INT_MAX-1))? (INT_MAX-1) : (temp+1);
            return FFO[index];
        }
    }
}

void FeatureExtraction(){
    int n = nodes.size(); // total number of nodes // PI
    // calculate lgfi
    cout << "=========================LGFI=========================" << endl;
    for (const auto& [id, name] : sorted_nodes){
        LGFI[name] = lgfi(name);
        cout << name << " => " << LGFI[name] << endl;
    }
    // calculate po
    
    initvector(PO, n); // fill in -1
    vector<int> new_po;
    show_current_id_node_map();
    cout << "=======PO=======" << endl;
    for (const auto& output_port: primary_output_node_name){
        cout << "Current_output:" << output_port << endl;
        new_po = po(output_port);
        update_if_smaller(PO, new_po);
        showvector(PO);
    }
    // calculate pi
    initvector(PI, n); // fill in -1
    cout << "=======PI========" << endl;
    vector<int> new_pi;
    for (const auto& input_port: primary_input_node_name){
        cout << "Current_input:" << input_port << endl;
        new_pi = pi(input_port);
        update_if_smaller(PI, new_pi);
        showvector(PI);
    }
    // calculate pi and ffo
    cout << "========================Start recursive part==========" << endl;
    int dummy;
    for (const auto& output_port: primary_output_node_name){
        dummy = ffo(output_port);
    }
    for (const auto& ff_input: dff_input_name){
        if(valid(ff_input)){
            dummy = ffo(wire_to_node[ff_input]);
        }
    }
    cout << "Finish FFO calculating" << endl;
    for (const auto& input_port: primary_input_node_name){
        cout << "Input port: " << input_port << endl;
        dummy = ffi(input_port);
    }
    for (const auto& ff_output: dff_node_name){
        dummy = ffi(ff_output);
    }
    cout << "Finish FFI calculating" << endl;
    process_FFI();
    process_FFO();
    cout << "=========================FFO=========================" << endl;
    for (const auto& [key, value] : FFO) {
        cout << key << " => " << value << endl;
    }
    cout << "=========================FFI=========================" << endl;
    for (const auto& [key, value] : FFI) {
        cout << key << " => " << value << endl;
    }
    /*
    cout << "========================Output dependency============" << endl;
    for (const auto& [name , to] : output_data_dependency){
        cout << name << ":" << endl;
        cout << "==offspring==" << endl;
        for(const auto& offspring : to){
            cout << offspring << endl;
        }
        cout << "==ancestor==" << endl;
        for(const auto& ancestor: nodes[name].inputs){
            if(valid(ancestor)) cout << wire_to_node[ancestor] << endl;
        }
        cout << "end" << endl;
    }
    */
}

void initialize(){
    // Primitive gates (and, or, nand, nor, not, buf, xor, xnor) and dff
    type_to_index["input"] = 0;
    type_to_index["output"] = 1;
    type_to_index["and"] = 2;
    type_to_index["or"] = 3;
    type_to_index["nand"] = 4;
    type_to_index["nor"] = 5;
    type_to_index["not"] = 6;
    type_to_index["buf"] = 7;
    type_to_index["xor"] = 8;
    type_to_index["xnor"] = 9;
    type_to_index["dff"] = 10;
    num_types = type_to_index.size();
    num_features = 5;
}

void start_and_clear(){

    FFI.clear();
    FFO.clear();
    PI.clear();
    PO.clear();
    LGFI.clear();
    nodes.clear();
    wire_to_node.clear();
    node_id_map.clear();
    id_node_map.clear();
    output_data_dependency.clear();
    primary_input_node_name.clear();
    primary_output_node_name.clear();
    dff_node_name.clear();
    dff_input_name.clear();
    sorted_nodes.clear();
    dff_dq_map.clear();
    Trojaned.clear();
}

string pad2(int i) {
    return (i < 10 ? "0" : "") + to_string(i);
}

int main(){

    bool svm_need_label = true;

    string output_base = "./training_data_for_svm/";
    fs::create_directories(output_base);

for(int j = 0; j <= 19; j++){

    initialize();

    ifstream infile("../release_all/trojan/design"+to_string(j)+".v");
    ifstream inresult("../release_all/trojan/result"+to_string(j)+".txt");
    //ifstream infile("circuit.v");
    ofstream edgefile("edges.csv");
    ofstream nodefile("nodetypes.csv");
    ofstream GNNedgefile("GNNedges.csv");
    ofstream GNNnodefile("GNNnodetypes.csv");
    ofstream GNNfeaturefile(output_base+"GNNfeature"+to_string(j)+".csv");

    string output_info_base = output_base + "info/design" + pad2(j) + "/";
    fs::create_directories(output_info_base);

    string line;
    int node_id_counter = 0;
    int line_count = 0;

    start_and_clear();

    while(getline(infile, line)){

        line_count ++;
        cout << line_count << endl;
        stringstream ss(line);
        string type; // can be wire, input, output, or node name?
        ss >> type;

        if(type == "and" || type == "or" || type == "nand" || type == "nor" || type == "xor" || type == "xnor" || type == "buf" || type == "not"){
            // type gx(a, b, c)
            string rest;
            getline(ss, rest);
            // rest = gx(a,b,c)
            size_t lpar = rest.find('(');
            size_t rpar = rest.find(')');
            string args = rest.substr(lpar + 1, rpar - lpar - 1);

            string node_name = rest.substr(0, lpar);
            node_name.erase(remove(node_name.begin(), node_name.end(), ' '), node_name.end());
            // node name now = gx
            // args contains the content: a, b, c
            vector<string> tokens;
            stringstream argss(args);
            string token;
            while (getline(argss, token, ',')) {
                token.erase(remove(token.begin(), token.end(), ' '), token.end());  // trim space
                tokens.push_back(token);
            }
            // tokens = [a, b, c]
            Node g;
            g.name = node_name;
            g.type = type;
            if(valid(tokens[0])) g.output = tokens[0];
            g.inputs = vector<string>(tokens.begin() + 1, tokens.end());
            // three hash tables!
            nodes[node_name] = g;
            if(!g.output.empty()) wire_to_node[g.output] = node_name; // therefore, the node is equivalent to output
            node_id_map[node_name] = node_id_counter++;
        }
        else if(type == "dff"){
            // TODO:
            // type dff name(.RN(1'b1), .SN(1'b1), .CK(n0), .D(n2480), .Q(n5[23]));
            string rest;
            getline(ss, rest);
            size_t lpar = rest.find('(');
            size_t endpos = rest.size();
            string args = rest.substr(lpar + 1, endpos - lpar - 1);
            
            string node_name = rest.substr(0, lpar);
            node_name.erase(remove(node_name.begin(), node_name.end(), ' '), node_name.end());

            // args = .RN(1'b1), .SN(1'b1), .CK(n0), .D(n2480), .Q(n5[23])
            vector<string> tokens;
            tokens = extract_connections(args);
            //for(const auto& k: tokens) cout << k << endl;
            // tokens = [1'b1, 1'b1, n0, n2480, n5[23]]
            Node dff;
            dff.name = node_name;
            dff.type = type;
            if(valid(tokens[4])) dff.output = tokens[4];
            dff_dq_map[node_name] = tokens[3];
            dff.inputs = vector<string>(tokens.begin(), tokens.end()-1);
            dff_input_name.insert(dff_input_name.end(), tokens.begin(), tokens.end()-1);
            // three hash tables!
            nodes[node_name] = dff;
            if(!dff.output.empty()) wire_to_node[dff.output] = node_name; // therefore, the node is equivalent to output
            node_id_map[node_name] = node_id_counter++;
            dff_node_name.push_back(node_name);
        }
        else if(type == "input"){
            string rest;
            getline(ss, rest);
            vector<string> tokens;
            tokens = expand_ports(rest);
            for(const auto& port_name : tokens){
                Node p;
                p.name = port_name;
                p.type = type;
                p.output = port_name;
                nodes[port_name] = p;
                wire_to_node[p.output] = port_name;
                node_id_map[port_name] = node_id_counter++;
                primary_input_node_name.push_back   (port_name);
            }
        }
        else if(type == "output"){
            string rest;
            getline(ss, rest);
            vector<string> tokens;
            tokens = expand_ports(rest);
            for(const auto& port_name : tokens){
                Node p;
                p.name = port_name;
                p.type = type;
                p.inputs.push_back(port_name);
                nodes[port_name] = p;
                node_id_map[port_name] = node_id_counter++;
                primary_output_node_name.push_back(port_name);
            }
        }
        else{

        }
    }

    for (const auto& [name, id] : node_id_map) {
        id_node_map[id] = name;
        sorted_nodes.emplace_back(id, name);
    }
    sort(sorted_nodes.begin(), sorted_nodes.end()); // sort with id

    // header
    GNNnodefile << "id,name";
    for (int i = 0; i < num_types; ++i) {
        GNNnodefile << ",type_" << i;
    }
    GNNnodefile << endl;
    for (const auto& [id, name] : sorted_nodes) {
        const string& type = nodes[name].type;
        //cout << type << endl;
        int type_id = type_to_index[type];
        //cout << type_id << endl;
        GNNnodefile << id;
        GNNnodefile << "," << name;
        for (int i = 0; i < num_types; ++i) {
            GNNnodefile << (i == type_id ? ",1" : ",0");
        }
        GNNnodefile << endl;
    }

    GNNedgefile << "source,target" << endl;
    for (const auto& [name, g] : nodes) {
        for (const string& input_wire : g.inputs) {
            if(valid(input_wire)){
                // iterate all g's input
                if (wire_to_node.find(input_wire) != wire_to_node.end()) {
                    string src_node = wire_to_node[input_wire];
                    GNNedgefile << node_id_map[src_node] << "," << node_id_map[name] << endl;
                    //edgefile << src_node << " -> " << name << endl;
                }
            }
        }
    }

    // not for python to read, for debug
    // output the types of node
    for(const auto& [name, g] : nodes){
        //nodefile << node_id_map[name] << ", " << g.type << endl;
        nodefile << "Name: " << name << ", " <<  node_id_map[name] << ", " << g.type << endl;
    }

    // construct edges
    for (const auto& [name, g] : nodes) {
        for (const string& input_wire : g.inputs) {
            if(valid(input_wire)){
                // iterate all g's input
                output_data_dependency[wire_to_node[input_wire]].push_back(name);
                if (wire_to_node.find(input_wire) != wire_to_node.end()) {
                    string src_node = wire_to_node[input_wire];
                    //edgefile << node_id_map[src_node] << "," << node_id_map[name] << endl;
                    edgefile << src_node << " -> " << name << endl;
                }
            }
        }
    }

    /*
    unordered_map<string, Node> nodes;                   // node_name -> Node
    unordered_map<string, string> wire_to_node;          // wire_name -> node_name
    unordered_map<string, int> node_id_map;              // node_name -> id
    
    vector<string> primary_input_node_name;
    vector<string> primary_output_node_name;
    vector<string> dff_node_name;
    */

    FeatureExtraction();
    // header
    GNNfeaturefile << "id,name,LGFi,FFi,FFo,Pi,Po";
    if(svm_need_label){
        read_trojanned_label(inresult);
        GNNfeaturefile << ",Trojan_gate";
    }
    GNNfeaturefile << endl;
    for (const auto& [id, name] : sorted_nodes) {
        GNNfeaturefile << id << "," << name << "," << LGFI[name] << "," << FFI[name] << "," << FFO[name] << "," << PI[id] << "," << PO[id];
        if(svm_need_label){
            GNNfeaturefile << "," << (Trojaned[id])?1:0;
        }
        GNNfeaturefile << endl;
    }
    save_nodes(output_info_base);
    save_graph_info(output_info_base);

} // end of each iteraiton

    return 0;
}