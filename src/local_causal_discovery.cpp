#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <filesystem>

using namespace std;

double correlation(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
    double sum_x2 = 0.0, sum_y2 = 0.0;

    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }

    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = sqrt((n * sum_x2 - sum_x * sum_x) *
                              (n * sum_y2 - sum_y * sum_y));

    if (denominator == 0.0) return 0.0;
    return numerator / denominator;
}

vector<string> split_csv_line(const string& line) {
    vector<string> result;
    string value;
    stringstream ss(line);

    while (getline(ss, value, ',')) {
        result.push_back(value);
    }

    return result;
}

string dataset_tag_from_path(const string& path) {
    size_t slash = path.find_last_of("/\\");
    string filename = (slash == string::npos) ? path : path.substr(slash + 1);

    size_t dot = filename.find_last_of('.');
    if (dot != string::npos) {
        filename = filename.substr(0, dot);
    }

    return filename;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <dataset_csv_path>\n";
        return 1;
    }

    string dataset_path = argv[1];
    string dataset_tag = dataset_tag_from_path(dataset_path);

    auto start = chrono::high_resolution_clock::now();

    ifstream file(dataset_path);
    if (!file.is_open()) {
        cerr << "Error: could not open " << dataset_path << "\n";
        return 1;
    }

    string line;
    vector<string> headers;
    vector<vector<double>> data;

    if (getline(file, line)) {
        headers = split_csv_line(line);
    }

    while (getline(file, line)) {
        vector<string> tokens = split_csv_line(line);
        vector<double> row;

        for (const auto& token : tokens) {
            try {
                row.push_back(stod(token));
            } catch (...) {
                row.push_back(0.0);
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();

    if (data.empty()) {
        cerr << "Error: dataset is empty or not parsed correctly.\n";
        return 1;
    }

    int rows = data.size();
    int cols = data[0].size();
    double threshold = 0.3;

    filesystem::create_directories("results/local_dags");
    filesystem::create_directories("results/metrics");
    filesystem::create_directories("results/plots");

    string edge_file   = "results/local_dags/" + dataset_tag + "_local_dag_edges.txt";
    string metric_file = "results/metrics/" + dataset_tag + "_local_metrics.txt";
    string dot_file    = "results/local_dags/" + dataset_tag + "_local_dag.dot";

    ofstream edge_out(edge_file);
    ofstream metric_out(metric_file);
    ofstream dot_out(dot_file);

    if (!edge_out.is_open() || !metric_out.is_open() || !dot_out.is_open()) {
        cerr << "Error: could not open output files.\n";
        return 1;
    }

    cout << "Dataset: " << dataset_tag << "\n";
    cout << "Rows: " << rows << "\n";
    cout << "Columns: " << cols << "\n\n";

    dot_out << "digraph " << dataset_tag << "_LocalDAG {\n";
    dot_out << "    rankdir=LR;\n";
    dot_out << "    bgcolor=\"white\";\n";
    dot_out << "    node [shape=ellipse, style=filled, fillcolor=lightblue, fontname=\"Helvetica\"];\n";
    dot_out << "    edge [color=gray, arrowsize=0.8];\n";

    int edge_count = 0;

    for (int i = 0; i < cols; i++) {
        for (int j = i + 1; j < cols; j++) {
            vector<double> xi, xj;

            for (int r = 0; r < rows; r++) {
                if (i < (int)data[r].size() && j < (int)data[r].size()) {
                    xi.push_back(data[r][i]);
                    xj.push_back(data[r][j]);
                }
            }

            if (xi.empty() || xj.empty()) continue;

            double corr = correlation(xi, xj);

            if (fabs(corr) >= threshold) {
                cout << headers[i] << " -> " << headers[j]
                     << "   corr = " << fixed << setprecision(4) << corr << "\n";

                edge_out << headers[i] << " -> " << headers[j]
                         << "   corr = " << fixed << setprecision(4) << corr << "\n";

                dot_out << "    \"" << headers[i] << "\" -> \"" << headers[j] << "\";\n";

                edge_count++;
            }
        }
    }

    dot_out << "}\n";

    auto end = chrono::high_resolution_clock::now();
    double exec_time = chrono::duration<double>(end - start).count();

    metric_out << "Dataset: " << dataset_tag << "\n";
    metric_out << "Rows: " << rows << "\n";
    metric_out << "Columns: " << cols << "\n";
    metric_out << "Threshold: " << threshold << "\n";
    metric_out << "Edges Detected: " << edge_count << "\n";
    metric_out << "Execution Time (sec): " << fixed << setprecision(6) << exec_time << "\n";

    edge_out.close();
    metric_out.close();
    dot_out.close();

    cout << "\nEdges detected: " << edge_count << "\n";
    cout << "Execution Time: " << fixed << setprecision(6) << exec_time << " seconds\n";
    cout << "Edge file: " << edge_file << "\n";
    cout << "Metric file: " << metric_file << "\n";
    cout << "DOT file: " << dot_file << "\n";

    return 0;
}
