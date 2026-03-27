import os
import re
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results/sample"
PLOTS_DIR = "results/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# 1. Correlation Heatmap
# -----------------------------
data_path = "data/prototype/california_housing.csv"
if not os.path.exists(data_path):
    data_path = "data/california_housing.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap - California Housing")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
    plt.close()

# -----------------------------
# 2. Chunk Edge Count Chart
# -----------------------------
summary_file = os.path.join(RESULTS_DIR, "chunk_summary.txt")
chunk_names = []
edge_counts = []

if os.path.exists(summary_file):
    with open(summary_file, "r") as f:
        for line in f:
            match = re.search(r"Chunk\s+(\d+).*edges detected\s*=\s*(\d+)", line)
            if match:
                chunk_names.append(f"Chunk {match.group(1)}")
                edge_counts.append(int(match.group(2)))

    if chunk_names:
        plt.figure(figsize=(8, 5))
        plt.bar(chunk_names, edge_counts)
        plt.title("Edges Detected per Chunk")
        plt.xlabel("Chunks")
        plt.ylabel("Edge Count")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "chunk_edge_counts.png"))
        plt.close()

# -----------------------------
# 3. Global Edge Frequency Chart
# -----------------------------
global_edges_file = os.path.join(RESULTS_DIR, "global_dag_edges.txt")
edge_labels = []
freqs = []

if os.path.exists(global_edges_file):
    with open(global_edges_file, "r") as f:
        for line in f:
            match = re.search(r"(.+?)\s+freq\s*=\s*(\d+)", line)
            if match:
                edge_labels.append(match.group(1).strip())
                freqs.append(int(match.group(2)))

    if edge_labels:
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(edge_labels)), freqs)
        plt.xticks(range(len(edge_labels)), edge_labels, rotation=75, ha="right")
        plt.title("Global DAG Edge Frequencies")
        plt.xlabel("Edges")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "edge_frequency_chart.png"))
        plt.close()

print("Visualizations generated in results/plots/")
