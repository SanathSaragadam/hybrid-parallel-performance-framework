#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>

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

int main() {
    auto start = chrono::high_resolution_clock::now();

    ifstream file("data/california_housing.csv");
    if (!file.is_open()) {
        cerr << "Error: could not open data/california_housing.csv\n";
        return 1;
    }

    string line;
    vector<string> headers;
    vector<vector<double>> data;

    // Read header
    if (getline(file, line)) {
        stringstream ss(line);
        string col;
        while (getline(ss, col, ',')) {
            headers.push_back(col);
        }
    }

    // Read rows
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> row;

        while (getline(ss, value, ',')) {
            try {
                row.push_back(stod(value));
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

    cout << "Rows: " << rows << "\n";
    cout << "Columns: " << cols << "\n\n";

    double threshold = 0.6;

    ofstream edge_out("results/sample/local_dag_edges.txt");
    ofstream metric_out("results/sample/local_dag_metrics.txt");
ofstream dot_out("results/sample/local_dag.dot");

    if (!edge_out.is_open() || !metric_out.is_open() || !dot_out.is_open()) {
    cerr << "Error: could not open output files.\n";
    return 1;
}

    cout << "Detected Causal Edges:\n";

    int edge_count = 0;
dot_out << "digraph CausalGraph {\n";
dot_out << "    rankdir=LR;\n";
dot_out << "    node [shape=box, style=filled, fillcolor=lightblue];\n";

    for (int i = 0; i < cols; i++) {
        for (int j = i + 1; j < cols; j++) {
            vector<double> xi, xj;

            for (int r = 0; r < rows; r++) {
                xi.push_back(data[r][i]);
                xj.push_back(data[r][j]);
            }

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

    auto end = chrono::high_resolution_clock::now();
    double exec_time = chrono::duration<double>(end - start).count();

    metric_out << "Rows: " << rows << "\n";
    metric_out << "Columns: " << cols << "\n";
    metric_out << "Threshold: " << threshold << "\n";
    metric_out << "Edges Detected: " << edge_count << "\n";
    metric_out << "Execution Time (sec): " << fixed << setprecision(6) << exec_time << "\n";

dot_out << "}\n"; 
   edge_out.close();
    metric_out.close();
dot_out.close();

    cout << "\nEdges detected: " << edge_count << "\n";
    cout << "Execution Time: " << fixed << setprecision(6) << exec_time << " seconds\n";
    cout << "Outputs written to results/sample/\n";

    return 0;
}
