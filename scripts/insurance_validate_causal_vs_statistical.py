import os
import re
import pandas as pd
import matplotlib.pyplot as plt

DATASET = "data/main/insurance.csv"
GLOBAL_EDGES = "results/global/insurance_global_dag_edges.txt"
REPORT_FILE = "results/metrics/insurance_validation_report.txt"
PLOTS_DIR = "results/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

# -----------------------------
# Load and preprocess dataset
# -----------------------------
df = pd.read_csv(DATASET)

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category").cat.codes

corr = df.corr(numeric_only=True)

# -----------------------------
# Statistical top influences on charges
# -----------------------------
stat_scores = {}
if "charges" in corr.columns:
    charges_corr = corr["charges"].drop("charges")
    for feature, value in charges_corr.items():
        stat_scores[feature] = abs(value)

# -----------------------------
# DAG edges pointing to charges
# -----------------------------
dag_scores = {}

if os.path.exists(GLOBAL_EDGES):
    with open(GLOBAL_EDGES, "r") as f:
        for line in f:
            line = line.strip()

            # format: left -> right   freq = N
            match = re.match(r"(.+?)\s*->\s*(.+?)\s+freq\s*=\s*(\d+)", line)
            if match:
                left = match.group(1).strip()
                right = match.group(2).strip()
                freq = int(match.group(3))

                if right == "charges":
                    dag_scores[left] = freq

# -----------------------------
# Create comparison dataframe
# -----------------------------
all_features = sorted(set(stat_scores.keys()) | set(dag_scores.keys()))

comparison = []
for feature in all_features:
    comparison.append({
        "feature": feature,
        "statistical_score": stat_scores.get(feature, 0.0),
        "dag_frequency": dag_scores.get(feature, 0)
    })

comp_df = pd.DataFrame(comparison)
comp_df = comp_df.sort_values(by=["statistical_score", "dag_frequency"], ascending=False)

# -----------------------------
# Save validation report
# -----------------------------
with open(REPORT_FILE, "w") as f:
    f.write("Insurance Dataset: Statistical vs Causal Validation\n\n")
    f.write("Top statistical influences on charges:\n")
    for feature, score in sorted(stat_scores.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{feature}: {score:.4f}\n")

    f.write("\nTop DAG edges into charges:\n")
    for feature, freq in sorted(dag_scores.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{feature} -> charges : freq = {freq}\n")

    f.write("\nComparison Table:\n")
    f.write(comp_df.to_string(index=False))

# -----------------------------
# Plot comparison
# -----------------------------
top_df = comp_df.head(6)

x = range(len(top_df))

plt.figure(figsize=(10, 5))
plt.bar([i - 0.2 for i in x], top_df["statistical_score"], width=0.4, label="Statistical Score")
plt.bar([i + 0.2 for i in x], top_df["dag_frequency"], width=0.4, label="DAG Frequency")

plt.xticks(list(x), top_df["feature"], rotation=45, ha="right")
plt.title("Statistical vs Causal Importance for Charges")
plt.xlabel("Features")
plt.ylabel("Score / Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "insurance_validation_comparison.png"))
plt.close()

print("Validation report and comparison chart generated.")
