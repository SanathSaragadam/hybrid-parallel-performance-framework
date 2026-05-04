import os
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from scipy.linalg import expm
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/hpc/job_table.parquet"

LOCAL_DIR = "results/notears/local_dags"
PLOT_DIR = "results/notears/plots"
METRIC_DIR = "results/notears/metrics"

os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)


# ============================================================
# NOTEARS CORE
# ============================================================

def _adj(w, d):
    return (w[: d * d] - w[d * d:]).reshape(d, d)


def _loss(W, X):
    R = X - X @ W
    return 0.5 / X.shape[0] * np.sum(R ** 2)


def _h(W):
    E = expm(W * W)
    return np.trace(E) - W.shape[0]


def notears_linear(X, lambda1=0.02, max_iter=12, h_tol=1e-8, rho_max=1e16, w_threshold=0.20):
    n, d = X.shape

    w_est = np.zeros(2 * d * d)
    rho = 1.0
    alpha = 0.0
    h_val = np.inf

    bounds = []
    for i in range(d):
        for j in range(d):
            if i == j:
                bounds.append((0, 0))
            else:
                bounds.append((0, None))
    bounds = bounds + bounds

    def objective(w):
        W = _adj(w, d)
        loss = _loss(W, X)
        h = _h(W)

        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * np.sum(w)

        E = expm(W * W)
        G_loss = -1.0 / n * X.T @ (X - X @ W)
        G_h = E.T * W * 2
        G_smooth = G_loss + (rho * h + alpha) * G_h

        grad_pos = G_smooth + lambda1
        grad_neg = -G_smooth + lambda1
        grad = np.concatenate([grad_pos.reshape(-1), grad_neg.reshape(-1)])

        return obj, grad

    for iteration in range(max_iter):
        while rho < rho_max:
            result = minimize(
                fun=lambda w: objective(w),
                x0=w_est,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 700}
            )

            w_new = result.x
            W_new = _adj(w_new, d)
            h_new = _h(W_new)

            if h_new > 0.25 * h_val:
                rho *= 10
            else:
                break

        w_est = w_new
        h_val = h_new
        alpha += rho * h_val

        if h_val <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est, d)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est, h_val


# ============================================================
# DATA PREPARATION
# ============================================================

def load_selected_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading PM100 dataset...")
    df = pd.read_parquet(DATA_PATH)

    numeric_df = df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    min_non_missing = int(0.75 * len(numeric_df))
    numeric_df = numeric_df.dropna(axis=1, thresh=min_non_missing)
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    nunique = numeric_df.nunique()
    numeric_df = numeric_df.loc[:, nunique > 1]

    variances = numeric_df.var().sort_values(ascending=False)
    selected_cols = variances.head(12).index.tolist()

    print("\nSelected features:")
    for col in selected_cols:
        print("-", col)

    selected_df = numeric_df[selected_cols].copy()

    # Keep chunk experiment manageable for laptop
    if len(selected_df) > 20000:
        selected_df = selected_df.sample(n=20000, random_state=42)

    selected_df = selected_df.reset_index(drop=True)

    return selected_df, selected_cols


# ============================================================
# VISUALIZATION
# ============================================================

def draw_local_dag(W, labels, output_path, chunk_id):
    G = nx.DiGraph()

    for label in labels:
        G.add_node(label)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if abs(W[i, j]) > 0:
                G.add_edge(labels[i], labels[j], weight=W[i, j])

    plt.figure(figsize=(14, 9))
    ax = plt.gca()
    ax.set_facecolor("#08111f")

    if len(G.edges()) == 0:
        plt.text(
            0.5,
            0.5,
            f"No edges found for chunk {chunk_id}",
            ha="center",
            va="center",
            fontsize=18,
            color="white"
        )
        plt.axis("off")
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#08111f")
        plt.close()
        return

    pos = nx.circular_layout(G)

    node_colors = []
    node_sizes = []

    for node in G.nodes():
        lower = node.lower()
        if any(k in lower for k in ["power", "energy", "time", "duration", "runtime", "elapsed"]):
            node_colors.append("#ff7b00")
            node_sizes.append(2500)
        else:
            node_colors.append("#00b4d8")
            node_sizes.append(1800)

    weights = [abs(G[u][v]["weight"]) for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0

    edge_widths = [
        1.2 + 4.5 * abs(G[u][v]["weight"]) / max_w
        for u, v in G.edges()
    ]

    edge_colors = [
        "#ffd166" if G[u][v]["weight"] > 0 else "#b5179e"
        for u, v in G.edges()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white",
        linewidths=2
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=22,
        edge_color=edge_colors,
        width=edge_widths,
        connectionstyle="arc3,rad=0.18",
        alpha=0.88
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
        font_weight="bold",
        font_color="black"
    )

    edge_labels = {
        (u, v): f"{G[u][v]['weight']:.2f}"
        for u, v in G.edges()
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color="white",
        bbox=dict(facecolor="#111827", edgecolor="none", alpha=0.75)
    )

    plt.title(
        f"Local NOTEARS DAG - Chunk {chunk_id}",
        fontsize=19,
        fontweight="bold",
        color="white",
        pad=20
    )

    plt.text(
        0.01,
        0.01,
        "Local DAG learned independently from one dataset partition",
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        bbox=dict(facecolor="#111827", edgecolor="#00b4d8", alpha=0.85, boxstyle="round,pad=0.5")
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#08111f")
    plt.close()


def save_edges(W, labels, path):
    with open(path, "w", encoding="utf-8") as f:
        count = 0
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if abs(W[i, j]) > 0:
                    count += 1
                    f.write(f"{labels[i]} -> {labels[j]} | weight = {W[i, j]:.4f}\n")

        if count == 0:
            f.write("No edges found.\n")


# ============================================================
# MAIN
# ============================================================

def main():
    chunk_size = 4000

    df, labels = load_selected_data()

    total_rows = len(df)
    num_chunks = int(np.ceil(total_rows / chunk_size))

    print(f"\nTotal rows used: {total_rows}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of chunks: {num_chunks}")

    summary_path = os.path.join(METRIC_DIR, "pm100_chunk_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as summary:
        summary.write("PM100 Chunked NOTEARS Summary\n")
        summary.write("============================\n\n")
        summary.write(f"Rows used: {total_rows}\n")
        summary.write(f"Chunk size: {chunk_size}\n")
        summary.write(f"Number of chunks: {num_chunks}\n\n")
        summary.write("Selected features:\n")
        for col in labels:
            summary.write(f"- {col}\n")
        summary.write("\nChunk Results:\n")

        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            end = min(start + chunk_size, total_rows)

            chunk_df = df.iloc[start:end].copy()

            if len(chunk_df) < 100:
                continue

            scaler = StandardScaler()
            X = scaler.fit_transform(chunk_df)

            print(f"\nRunning NOTEARS on chunk {chunk_id}: rows {start} to {end}")

            t0 = time.time()
            W, h_val = notears_linear(
                X,
                lambda1=0.02,
                max_iter=12,
                w_threshold=0.20
            )
            runtime = time.time() - t0

            edge_count = int(np.sum(np.abs(W) > 0))

            edge_file = os.path.join(LOCAL_DIR, f"pm100_chunk_{chunk_id}_edges.txt")
            adj_file = os.path.join(LOCAL_DIR, f"pm100_chunk_{chunk_id}_adjacency.csv")
            plot_file = os.path.join(PLOT_DIR, f"pm100_local_dag_chunk_{chunk_id}.png")

            save_edges(W, labels, edge_file)
            pd.DataFrame(W, index=labels, columns=labels).to_csv(adj_file)
            draw_local_dag(W, labels, plot_file, chunk_id)

            summary.write(
                f"Chunk {chunk_id}: rows={start}-{end}, "
                f"edges={edge_count}, runtime_sec={runtime:.4f}, h={h_val:.10f}\n"
            )

            print(f"Chunk {chunk_id} done | edges={edge_count} | runtime={runtime:.2f}s")

    print("\nDONE.")
    print("Saved chunk summary:", summary_path)
    print("Saved local DAGs in:", PLOT_DIR)


if __name__ == "__main__":
    main()
