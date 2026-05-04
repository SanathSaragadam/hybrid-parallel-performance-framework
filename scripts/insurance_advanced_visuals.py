import os
import re
import pandas as pd
import matplotlib.pyplot as plt

DATASET = "data/main/insurance.csv"
GLOBAL_EDGES = "results/global/insurance_global_dag_edges.txt"
CHUNK_SUMMARY = "results/metrics/insurance_chunk_summary.txt"
PLOTS_DIR = "results/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATASET)

# Convert categorical columns to numeric where needed
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category").cat.codes

# -----------------------------
# 1. Correlation Heatmap
# -----------------------------
corr = df.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Insurance Dataset Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "insurance_correlation_heatmap.png"))
plt.close()

# -----------------------------
# 2. Charges Influence Chart
# -----------------------------
if "charges" in corr.columns:
    charges_corr = corr["charges"].drop("charges").sort_values(key=lambda x: abs(x), ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(charges_corr.index, charges_corr.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature Influence on Charges")
    plt.xlabel("Features")
    plt.ylabel("Correlation with Charges")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "insurance_charges_influence_chart.png"))
    plt.close()

# -----------------------------
# 3. Global Edge Frequency Chart
# -----------------------------
edge_labels = []
freqs = []

if os.path.exists(GLOBAL_EDGES):
    with open(GLOBAL_EDGES, "r") as f:
        for line in f:
            match = re.search(r"(.+?)\s+freq\s*=\s*(\d+)", line)
            if match:
                edge_labels.append(match.group(1).strip())
                freqs.append(int(match.group(2)))

if edge_labels:
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(edge_labels)), freqs)
    plt.xticks(range(len(edge_labels)), edge_labels, rotation=60, ha="right")
    plt.title("Insurance Global DAG Edge Frequencies")
    plt.xlabel("Edges")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "insurance_edge_frequency_chart.png"))
    plt.close()

# -----------------------------
# 4. Chunk Edge Count Chart
# -----------------------------
chunk_names = []
edge_counts = []

if os.path.exists(CHUNK_SUMMARY):
    with open(CHUNK_SUMMARY, "r") as f:
        for line in f:
            match = re.search(r"Chunk\s+(\d+).*edges detected\s*=\s*(\d+)", line)
            if match:
                chunk_names.append(f"Chunk {match.group(1)}")
                edge_counts.append(int(match.group(2)))

if chunk_names:
    plt.figure(figsize=(8, 5))
    plt.bar(chunk_names, edge_counts)
    plt.title("Insurance Chunk Edge Counts")
    plt.xlabel("Chunks")
    plt.ylabel("Edges Detected")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "insurance_chunk_edge_counts.png"))
    plt.close()

print("Insurance advanced visuals generated in results/plots/")
