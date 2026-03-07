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

    if (getline(file, line)) {
        stringstream ss(line);
        string col;
        while (getline(ss, col, ',')) {
            headers.push_back(col);
        }
    }

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

    int chunk_size = 5000;
    int num_chunks = (rows + chunk_size - 1) / chunk_size;
    double threshold = 0.6;

    cout << "Rows: " << rows << "\n";
    cout << "Columns: " << cols << "\n";
    cout << "Chunk size: " << chunk_size << "\n";
    cout << "Number of chunks: " << num_chunks << "\n\n";

    ofstream summary_out("results/sample/chunk_summary.txt");
    if (!summary_out.is_open()) {
        cerr << "Error: could not open chunk summary file.\n";
        return 1;
    }

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int start_row = chunk * chunk_size;
        int end_row = min(start_row + chunk_size, rows);

        string edge_filename = "results/sample/chunk_" + to_string(chunk) + "_edges.txt";
        string dot_filename  = "results/sample/chunk_" + to_string(chunk) + ".dot";

        ofstream edge_out(edge_filename);
        ofstream dot_out(dot_filename);

        if (!edge_out.is_open() || !dot_out.is_open()) {
            cerr << "Error: could not open output file for chunk " << chunk << "\n";
            return 1;
        }

        dot_out << "digraph Chunk" << chunk << " {\n";
        dot_out << "    rankdir=LR;\n";
        dot_out << "    node [shape=box, style=filled, fillcolor=lightyellow];\n";

        int edge_count = 0;

        for (int i = 0; i < cols; i++) {
            for (int j = i + 1; j < cols; j++) {
                vector<double> xi, xj;

                for (int r = start_row; r < end_row; r++) {
                    xi.push_back(data[r][i]);
                    xj.push_back(data[r][j]);
                }

                double corr = correlation(xi, xj);

                if (fabs(corr) >= threshold) {
                    edge_out << headers[i] << " -> " << headers[j]
                             << "   corr = " << fixed << setprecision(4) << corr << "\n";

                    dot_out << "    \"" << headers[i] << "\" -> \"" << headers[j] << "\";\n";

                    edge_count++;
                }
            }
        }

        dot_out << "}\n";
        edge_out.close();
        dot_out.close();

        summary_out << "Chunk " << chunk
                    << ": rows " << start_row << "-" << end_row - 1
                    << ", edges detected = " << edge_count << "\n";

        cout << "Chunk " << chunk << " completed, edges detected = " << edge_count << "\n";
    }

    auto end = chrono::high_resolution_clock::now();
    double exec_time = chrono::duration<double>(end - start).count();

    summary_out << "\nTotal execution time: " << fixed << setprecision(6) << exec_time << " sec\n";
    summary_out.close();

    cout << "\nChunk-based DAG generation complete.\n";
    cout << "Execution Time: " << fixed << setprecision(6) << exec_time << " seconds\n";

    return 0;
}
