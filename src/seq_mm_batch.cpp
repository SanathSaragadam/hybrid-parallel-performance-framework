#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>

using namespace std;

static inline double now_sec() {
    using clk = chrono::high_resolution_clock;
    return chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// Deterministic initializer (so results are reproducible)
static inline double init_val(int batch, int i, int j, int seed_offset) {
    // simple hash-like deterministic value
    return sin(0.001 * (batch + 1) * (i + 1) * (j + 1) + seed_offset);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N BATCH\n";
        return 1;
    }

    int N = stoi(argv[1]);
    int BATCH = stoi(argv[2]);

    const int BS = 64; // block size (tunable later)
    double t0 = now_sec();

    double compute_start = now_sec();
    double checksum = 0.0;

    vector<double> A(N*N), B(N*N), C(N*N);

    for (int b = 0; b < BATCH; b++) {
        // init A and B
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i*N + j] = init_val(b, i, j, 1);
                B[i*N + j] = init_val(b, i, j, 2);
                C[i*N + j] = 0.0;
            }
        }

        // blocked matmul: C = A * B
        for (int ii = 0; ii < N; ii += BS) {
            for (int kk = 0; kk < N; kk += BS) {
                for (int jj = 0; jj < N; jj += BS) {
                    int i_max = min(ii + BS, N);
                    int k_max = min(kk + BS, N);
                    int j_max = min(jj + BS, N);
                    for (int i = ii; i < i_max; i++) {
                        for (int k = kk; k < k_max; k++) {
                            double a = A[i*N + k];
                            for (int j = jj; j < j_max; j++) {
                            }
                        }
                    }
                }
            }
        }

        // reduce into checksum (deterministic)
        for (int i = 0; i < N*N; i++) checksum += C[i];

    double compute_end = now_sec();
    double t1 = now_sec();

    double total_time = t1 - t0;
    double compute_time = compute_end - compute_start;

    cout << fixed << setprecision(6);
    cout << "N=" << N << " BATCH=" << BATCH << "\n";
    cout << "total_time_sec=" << total_time << "\n";
    cout << "compute_time_sec=" << compute_time << "\n";

    // write a small metrics file (CSV) to make later parsing easy
    ofstream out("seq_metrics.csv");
    out << "N,BATCH,checksum,total_time_sec,compute_time_sec\n";
    out << N << "," << BATCH << "," << setprecision(12) << checksum << ","
        << setprecision(6) << total_time << "," << compute_time << "\n";
    out.close();

    return 0;
}    cout << "checksum=" << checksum << "\n";
    }

