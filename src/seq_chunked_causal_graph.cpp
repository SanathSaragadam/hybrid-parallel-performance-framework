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

    int chunk_size = 5000;
    int num_chunks = (rows + chunk_size - 1) / chunk_size;
    double threshold = 0.6;

    filesystem::create_directories("results/local_dags");
    filesystem::create_directories("results/metrics");

    cout << "Dataset: " << dataset_tag << "\n";
    cout << "Rows: " << rows << "\n";
    cout << "Columns: " << cols << "\n";
    cout << "Chunk size: " << chunk_size << "\n";
    cout << "Chunks: " << num_chunks << "\n\n";

    string summary_file = "results/metrics/" + dataset_tag + "_chunk_summary.txt";
    ofstream summary_out(summary_file);

    if (!summary_out.is_open()) {
        cerr << "Error: could not open summary output file.\n";
        return 1;
    }

    auto start = chrono::high_resolution_clock::now();

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int start_row = chunk * chunk_size;
        int end_row = min(start_row + chunk_size, rows);

        string edge_file =
            "results/local_dags/" + dataset_tag + "_chunk_" + to_string(chunk) + "_edges.txt";

        ofstream edge_out(edge_file);

        if (!edge_out.is_open()) {
            cerr << "Error: could not open edge file for chunk " << chunk << "\n";
            return 1;
        }

        int edge_count = 0;

        for (int i = 0; i < cols; i++) {
            for (int j = i + 1; j < cols; j++) {
                vector<double> xi, xj;

                for (int r = start_row; r < end_row; r++) {
                    if (i < (int)data[r].size() && j < (int)data[r].size()) {
                        xi.push_back(data[r][i]);
                        xj.push_back(data[r][j]);
                    }
                }

                if (xi.empty() || xj.empty()) continue;

                double corr = correlation(xi, xj);

                if (fabs(corr) >= threshold) {
                    edge_out << headers[i] << " -> " << headers[j]
                             << "   corr = " << fixed << setprecision(4) << corr << "\n";
                    edge_count++;
                }
            }
        }

        edge_out.close();

        summary_out << "Chunk " << chunk
                    << ": rows " << start_row << "-" << (end_row - 1)
                    << ", edges detected = " << edge_count << "\n";

        cout << "Chunk " << chunk
             << " completed, edges detected = " << edge_count << "\n";
    }

    auto end = chrono::high_resolution_clock::now();
    double exec_time = chrono::duration<double>(end - start).count();

    summary_out << "\nExecution Time (sec): "
                << fixed << setprecision(6) << exec_time << "\n";
    summary_out.close();

    cout << "\nChunked DAG complete for " << dataset_tag << "\n";
    cout << "Summary file: " << summary_file << "\n";
    cout << "Execution Time: " << fixed << setprecision(6) << exec_time << " seconds\n";

    return 0;
}
