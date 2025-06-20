#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;
// type to one hot encoding
unordered_map<string, int> type_to_index;
int num_types;

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
}

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

int main(){

    bool islabelled = true;
    const string base_input_path = "../release_all/trojan/";
    //const string base_input_path = "../WithoutLabelCases/release2/";
    
    const string base_output_path = "./dataset/";
    //const string base_output_path = "./testset/";
    initialize();

    for(int i = 10; i < 30; i ++){
    int label = i + 1;
    string design_dir = "design"+to_string(i);
    string design_file = design_dir+".v";
    string input_path = base_input_path+design_file;
    string output_path = base_output_path + "raw/" + design_dir + "/";
    
    fs::create_directories(output_path);

    ifstream infile(input_path);

    //ifstream infile("circuit.v");
    ofstream edgefile(output_path+"edges.csv");
    ofstream nodefile(output_path+"nodetypes.csv");
    ofstream GNNedgefile(output_path+"GNNedges.csv");
    ofstream GNNnodefile(output_path+"GNNnodetypes.csv");
    
    unordered_map<string, Node> nodes;                   // node_name -> Node
    unordered_map<string, string> wire_to_node;          // wire_name -> node_name
    unordered_map<string, int> node_id_map;              // node_name -> id
    
    string line;
    int node_id_counter = 0;
    int line_count = 0;

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
            dff.inputs = vector<string>(tokens.begin(), tokens.end()-1);
            // three hash tables!
            nodes[node_name] = dff;
            if(!dff.output.empty()) wire_to_node[dff.output] = node_name; // therefore, the node is equivalent to output
            node_id_map[node_name] = node_id_counter++;
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
            }
        }
        else{

        }
    }

    vector<pair<int, string>> sorted_nodes; // (id, name)
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
    // create label
    if(islabelled){
        ofstream labelfile(output_path+"label.txt");
        labelfile << label << endl;
    }
    
    }
    return 0;
}