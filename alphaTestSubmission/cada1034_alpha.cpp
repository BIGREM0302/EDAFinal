// cada1034_alpha.cpp
#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

int main(int argc, char* argv[]) {

    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " -netlist <netlist_path> -output <output_path>" << endl;
        return 1;
    }

    string netlist_path, output_path;
    string temp_result_path = "hasTrojan.txt";
    string input_gates_path = "filter_result.txt";

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-netlist") netlist_path = argv[++i];
        else if (arg == "-output") output_path = argv[++i];
    }

    ofstream output_file(output_path);

    // Prepare all data needed
    // produce ./parser_result, and ./parser_result/info
    string preprocess_cmd = "./parser " + netlist_path;
    if (system(preprocess_cmd.c_str()) != 0) {
        cerr << "Entrance::Preprocess failed." << endl;
        return 1;
    }

    // Execute phase1: GNN inference
    string python_phase1_cmd = "python3 gnn_inference.py";
    if (system(python_phase1_cmd.c_str()) != 0) {
        cerr << "Entrance::GNN analysis failed." << endl;
        return 1;
    }

    ifstream temp_result_file(temp_result_path);
    // see if there is trojan
    string hastrojan;
    temp_result_file >> hastrojan;

    if(hastrojan == "False"){
        cout << "Entrance::The netlist doesn't have trojan" << endl;
        output_file << "NO_TROJAN";
    }
    else if(hastrojan == "True"){
        cout << "Entrance::The netlist has trojan" << endl;
        output_file << "TROJANED" << endl << "TROJAN_GATES" << endl;

        string python_phase2_cmd = "python3 svm_infer.py";
        
        if (system(python_phase2_cmd.c_str()) != 0) {
            cerr << "Entrance::SVM inference failed." << endl;
            return 1;
        }

        string filter_cmd = "./connect_filter";
        if (system(filter_cmd.c_str()) != 0) {
            cerr << "Entrance::Connect filter fails to work" << endl;
            return 1;
        }        

        ifstream input_gates_file(input_gates_path);
        string line;
        while(getline(input_gates_file, line)){
            output_file << line << endl;
        }
        output_file << "END_TROJAN_GATES";
    }

    cout << "Analysis completed. Output: " << output_path << endl;
    return 0;
}
