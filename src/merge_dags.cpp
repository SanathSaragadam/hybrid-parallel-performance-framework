#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <iomanip>

using namespace std;

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
            continue; // skip missing chunk files
        }

        string line;
        while (getline(file, line)) {
            if (!line.empty()) {
                edge_frequency[line]++;
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
        string full_edge = entry.first;
        int freq = entry.second;

        // Keep edges appearing in at least 2 chunks
        if (freq >= 2) {
            edge_out << full_edge << "   freq = " << freq << "\n";
            cout << full_edge << "   freq = " << freq << "\n";

            // Extract only the edge part before "corr ="
            size_t corr_pos = full_edge.find("   corr");
            string edge_only = (corr_pos != string::npos) ? full_edge.substr(0, corr_pos) : full_edge;

            size_t arrow_pos = edge_only.find("->");
            if (arrow_pos != string::npos) {
                string left = edge_only.substr(0, arrow_pos);
                string right = edge_only.substr(arrow_pos + 2);

                // trim spaces
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
