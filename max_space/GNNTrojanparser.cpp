#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>

using namespace std;

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

void clean(string& ss){
    ss.erase(remove(ss.begin(), ss.end(), ' '), ss.end());
}

bool valid(const string& test){
    return (test != "1'b1" && test != "1'b0");
}

// To expand array like ports (only input or output need)
vector<string> expand_ports(const string& line, bool isPort = true) {
    vector<string> result;
    //cout << line << endl;
    smatch match;
    regex bit_range_regex(R"(\[(\d+):(\d+)\])");

    string range_part;
    string var_part;

    if (regex_search(line, match, bit_range_regex)) {
        int high = stoi(match[1]);
        int low = stoi(match[2]);
        //cout << high << " " << low << endl;
        range_part = match.str();
        if(isPort) var_part = line.substr(match.position() + match.length());
        else var_part = line.substr(0,match.position());
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
        clean(connection);
        //connection.erase(remove(connection.begin(), connection.end(), ' '), connection.end());
        results.push_back(connection);
    }

    return results;
}


vector<string> extract_items(const string& input) {
    size_t start = input.find('{');
    size_t end = input.find('}');
    string cleaned = input.substr(start + 1, end - start - 1);

    // 放結果的 vector
    vector<string> result;
    stringstream ss(cleaned);
    string item;

    while (getline(ss, item, ',')) {
        clean(item);
        result.push_back(item);
    }
    return result;
}

struct Node {
    string name;
    string type;
    string output;
    vector<string> inputs;
};

int main(){

    //ifstream infile("../release(20250517)/release/design9.v");
    //ifstream infile("circuit.v");
    ifstream infile("./Flatten_Trojans/trojan5_flatten.v");
    //ifstream infile("trojancircuit.v");
    ofstream edgefile("edges.csv");
    ofstream nodefile("nodetypes.csv");
    ofstream GNNedgefile("GNNedges.csv");
    ofstream GNNnodefile("GNNnodetypes.csv");

    unordered_map<string, Node> nodes;                   // node_name -> Node
    unordered_map<string, string> wire_to_node;          // wire_name -> node_name
    unordered_map<string, int> node_id_map;              // node_name -> id

    initialize();

    string line;
    int node_id_counter = 0;
    int line_count = 0;

    while(getline(infile, line)){
        line.erase(0, line.find_first_not_of(" \t"));
        line_count ++;
        cout << line_count << endl;

        stringstream ss(line);
        string type; // can be wire, input, output, or node name?
        ss >> type;

        if(type == "and" || type == "or" || type == "nand" || type == "nor" || type == "xor" || type == "xnor" || type == "buf" || type == "not"){
            // type gx(.A(a), .B(b), .Y(c));
            string rest;
            getline(ss, rest);

            while (std::getline(infile, line)) {
                line_count ++;
                cout << line_count << endl;
                rest += line;
                if (line.find(");") != std::string::npos) break;
            }

            size_t lpar = rest.find('(');
            size_t endpos = rest.size();
            string args = rest.substr(lpar + 1, endpos - lpar - 1);
            
            string node_name = rest.substr(0, lpar);
            clean(node_name);
            //node_name.erase(remove(node_name.begin(), node_name.end(), ' '), node_name.end());

            // args = .A(a), .B(b), .Y(c)
            vector<string> tokens;
            tokens = extract_connections(args);
            //for(const auto& k: tokens) cout << k << endl;
            // tokens = [a, b, c]
            Node g;
            g.name = node_name;
            g.type = type;
            if(valid(tokens[tokens.size()-1])) g.output = tokens[tokens.size()-1];
            g.inputs = vector<string>(tokens.begin(), tokens.end()-1);
            // three hash tables!
            nodes[node_name] = g;
            if(!g.output.empty()) wire_to_node[g.output] = node_name; // therefore, the node is equivalent to output
            node_id_map[node_name] = node_id_counter++;
        }
        else if(type == "dff"){
            // TODO:
            // type dff name(.C(clk),.D(_008_),.Q(counter[0] ),.RST(1'b0),.SET(1'b0));
            string rest;
            getline(ss, rest);

            while (std::getline(infile, line)) {
                line_count ++;
                cout << line_count << endl;
                rest += line;
                if (line.find(");") != std::string::npos) break;
            }

            size_t lpar = rest.find('(');
            size_t endpos = rest.size();
            string args = rest.substr(lpar + 1, endpos - lpar - 1);
            
            string node_name = rest.substr(0, lpar);
            clean(node_name);
            //node_name.erase(remove(node_name.begin(), node_name.end(), ' '), node_name.end());

            // args = .C(clk),.D(_008_),.Q(counter[0] ),.RST(1'b0),.SET(1'b0)
            vector<string> tokens;
            tokens = extract_connections(args);
            //for(const auto& k: tokens) cout << k << endl;
            // tokens = [clk, _008_, counter[0], 1'b0, 1'b0]
            Node dff;
            dff.name = node_name;
            dff.type = type;
            if(valid(tokens[2])) dff.output = tokens[2];
            dff.inputs.push_back(tokens[0]);
            dff.inputs.push_back(tokens[1]);
            dff.inputs.push_back(tokens[3]);
            dff.inputs.push_back(tokens[4]);
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
                //wire_to_node[p.output] = port_name; // will be replaced
                node_id_map[port_name] = node_id_counter++;
            }
        }
        else if(type == "assign"){
            // since assing only occurs in last serveral lines, rhs and lhs nodes both have been constructed
            string rest;
            getline(ss, rest);
            size_t pos = rest.find("=");
            string lhs, rhs;
            if (pos != string::npos) {
                lhs = rest.substr(0, pos);
                clean(lhs);
                //cout << "lhs = " << lhs << endl;
                rhs = rest.substr(pos + 1);
                clean(rhs);
                //cout << "rhs = " << rhs << endl;
            } else {
                cout << "Can't find = in assignment" << endl;
            }
            //lhs
            vector<string> ltokens;
            ltokens = expand_ports(lhs,false);
            //rhs
            vector<string> rtokens;
            if(ltokens.size() <= 1){
                size_t scend = rhs.find(";");
                //cout << rhs.substr(0,scend) << endl;
                rtokens.push_back(rhs.substr(0,scend));
            }
            else if(ltokens.size() > 1){
                rtokens = extract_items(rhs);
            }
            if(ltokens.size() != rtokens.size())
            cout << "Something error occurs in assignment read: " << ltokens.size() << "!=" << rtokens.size() << endl;
            for(int i = 0; i < ltokens.size(); i++){
                wire_to_node[ltokens[i]] = wire_to_node[rtokens[i]];
            }
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

    return 0;
}