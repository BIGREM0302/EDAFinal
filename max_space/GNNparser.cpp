#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>
#include <climits>

using namespace std;

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
unordered_map<string, int> PI;
unordered_map<string, int> PO;
unordered_map<string, int> LGFI;

unordered_map<string, Node> nodes;                   // node_name -> Node
unordered_map<string, string> wire_to_node;          // wire_name -> node_name
unordered_map<string, int> node_id_map;              // node_name -> id
unordered_map<string, string> dff_dq_map;           // d <-> q

vector<string> primary_input_node_name;
vector<string> primary_output_node_name;
vector<string> dff_node_name;
vector<string> dff_input_name;

vector<pair<int, string>> sorted_nodes; // (id, name)

int lgfi(string index){
    if(nodes[index].inputs.size() <= 0){
        return 1;
    }
    else {
        int temp = 0;
        for(const auto& ancestor:nodes[index].inputs){
            //cout << ancestor << ":" << nodes[ancestor].inputs.size() << endl;
            temp += (nodes[wire_to_node[ancestor]].inputs.size()<=0)?1:nodes[wire_to_node[ancestor]].inputs.size();
        }
        return temp;
    }
}

int pi(string index){
    if(PI.count(index) > 0){
        return PI[index];
    }
    else if(nodes[index].type == "input"){
        PI[index] = 0;
        return PI[index];
    }
    else if(nodes[index].type == "dff"){
        PI[index] = INT_MAX; // fear of cycle
        return PI[index];
    }
    else{
       if(nodes[index].inputs.size() <= 0){
            PI[index] = INT_MAX;
            cout << "Connect to a constant" << endl;
            return PI[index];
        }
        else{
            int temp = INT_MAX;
            for(const auto& ancestor : nodes[index].inputs){
                if(pi(wire_to_node[ancestor]) < temp) temp = PI[wire_to_node[ancestor]];
            }
            PI[index] = (temp == INT_MAX)? INT_MAX : (temp+1);
            return PI[index];
        } 
    }
}

int po(){}

int ffi(string index){}

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
            FFO[index] = INT_MAX;
            return FFO[index];
        }
        else{
            int temp = INT_MAX;
            for(const auto& ancestor : nodes[index].inputs){
                if(ffo(wire_to_node[ancestor]) < temp) temp = FFO[wire_to_node[ancestor]];
            }
            FFO[index] = (temp == INT_MAX)? INT_MAX : (temp+1);
            return FFO[index];
        }
    }
}

void FeatureExtraction(){
    // calculate lgfi
    cout << "=========================LGFI=========================" << endl;
    for (const auto& [id, name] : sorted_nodes){
        LGFI[name] = lgfi(name);
        cout << name << " => " << LGFI[name] << endl;
    }
    // calculate pi and ffo
    cout << "Start recursive part" << endl;
    int dummy;
    for (const auto& output_port: primary_output_node_name){
        dummy = pi(output_port);
        dummy = ffo(output_port);
    }
    for (const auto& ff_input: dff_input_name){
        if(valid(ff_input)){
            dummy = pi(wire_to_node[ff_input]);
            dummy = ffo(wire_to_node[ff_input]);
        }
    }
    for (const auto& ff_output: dff_node_name){
        PI[ff_output] = 1 + PI[wire_to_node[dff_dq_map[ff_output]]];
        FFO[ff_output] = 1 + FFO[wire_to_node[dff_dq_map[ff_output]]];
    }
    cout << "=========================PI=========================" << endl;
    for (const auto& [key, value] : PI) {
        cout << key << " => " << value << endl;
    }
    cout << "=========================FFO=========================" << endl;
    for (const auto& [key, value] : FFO) {
        cout << key << " => " << value << endl;
    }
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
    num_features = 3;
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
    primary_input_node_name.clear();
    primary_output_node_name.clear();
    dff_node_name.clear();
    dff_input_name.clear();
    sorted_nodes.clear();
    dff_dq_map.clear();
}


int main(){

    ifstream infile("../release(20250517)/release/design0.v");
    //ifstream infile("circuit.v");
    ofstream edgefile("edges.csv");
    ofstream nodefile("nodetypes.csv");
    ofstream GNNedgefile("GNNedges.csv");
    ofstream GNNnodefile("GNNnodetypes.csv");
    ofstream GNNfeaturefile("GNNfeature.csv");

    initialize();

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
                primary_input_node_name.push_back(port_name);
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
    GNNfeaturefile << "id,name";
    for (int i = 0; i < num_features; ++i) {
        GNNfeaturefile << ",feature_" << i;
    }
    GNNfeaturefile << endl;
    for (const auto& [id, name] : sorted_nodes) {
        GNNfeaturefile << id << "," << name << "," << PI[name] << "," << FFO[name] << "," << LGFI[name] << endl;
    }

    return 0;
}