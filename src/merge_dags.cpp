#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <iomanip>

using namespace std;

string extract_edge_only(const string& line) {
    size_t corr_pos = line.find("   corr");
    if (corr_pos != string::npos) {
        return line.substr(0, corr_pos);
    }
    return line;
}

int main() {
    vector<string> chunk_files = {
        "results/sample/chunk_0_edges.txt",
        "results/sample/chunk_1_edges.txt",
        "results/sample/chunk_2_edges.txt",
        "results/sample/chunk_3_edges.txt",
        "results/sample/chunk_4_edges.txt"
    };

    map<string, int> edge_frequency;

    for (const string& filename : chunk_files) {
        ifstream file(filename);
        if (!file.is_open()) {
            continue;
        }

        string line;
        while (getline(file, line)) {
            if (!line.empty()) {
                string edge_only = extract_edge_only(line);
                edge_frequency[edge_only]++;
            }
        }

        file.close();
    }

    ofstream edge_out("results/sample/global_dag_edges.txt");
    ofstream dot_out("results/sample/global_dag.dot");

    if (!edge_out.is_open() || !dot_out.is_open()) {
        cerr << "Error: could not open global DAG output files.\n";
        return 1;
    }

    dot_out << "digraph GlobalCausalGraph {\n";
    dot_out << "    rankdir=LR;\n";
    dot_out << "    node [shape=box, style=filled, fillcolor=lightgreen];\n";

    int global_edge_count = 0;

    cout << "Merged Global DAG Edges:\n";

    for (const auto& entry : edge_frequency) {
        string edge_only = entry.first;
        int freq = entry.second;

        if (freq >= 2) {
            edge_out << edge_only << "   freq = " << freq << "\n";
            cout << edge_only << "   freq = " << freq << "\n";

            size_t arrow_pos = edge_only.find("->");
            if (arrow_pos != string::npos) {
                string left = edge_only.substr(0, arrow_pos);
                string right = edge_only.substr(arrow_pos + 2);

                while (!left.empty() && left.front() == ' ') left.erase(0, 1);
                while (!left.empty() && left.back() == ' ') left.pop_back();
                while (!right.empty() && right.front() == ' ') right.erase(0, 1);
                while (!right.empty() && right.back() == ' ') right.pop_back();

                dot_out << "    \"" << left << "\" -> \"" << right << "\";\n";
                global_edge_count++;
            }
        }
    }

    dot_out << "}\n";

    edge_out.close();
    dot_out.close();

    cout << "\nGlobal DAG edge count: " << global_edge_count << "\n";
    cout << "Outputs written to results/sample/global_dag_edges.txt and global_dag.dot\n";

    return 0;
}
