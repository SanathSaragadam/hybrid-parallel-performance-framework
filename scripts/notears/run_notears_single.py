import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from scipy.linalg import expm
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/hpc/job_table.parquet"

OUT_DIR = "results/notears"
PLOT_DIR = "results/notears/plots"
GLOBAL_DIR = "results/notears/global"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(GLOBAL_DIR, exist_ok=True)


# ============================================================
# NOTEARS CORE IMPLEMENTATION
# ============================================================

def _adj(w, d):
    """
    Convert optimization vector into weighted adjacency matrix.
    NOTEARS represents positive and negative weights separately.
    """
    return (w[: d * d] - w[d * d:]).reshape(d, d)


def _loss(W, X):
    """
    Least squares reconstruction loss.
    """
    M = X @ W
    R = X - M
    return 0.5 / X.shape[0] * np.sum(R ** 2)


def _h(W):
    """
    NOTEARS acyclicity constraint:
    h(W) = trace(exp(W o W)) - d
    If h(W) = 0, the graph is acyclic.
    """
    E = expm(W * W)
    return np.trace(E) - W.shape[0]


def notears_linear(
    X,
    lambda1=0.02,
    max_iter=20,
    h_tol=1e-8,
    rho_max=1e16,
    w_threshold=0.20
):
    """
    Simplified NOTEARS linear model.

    X: standardized data matrix
    lambda1: sparsity penalty
    w_threshold: removes weak edges after optimization
    """
    n, d = X.shape

    w_est = np.zeros(2 * d * d)
    rho = 1.0
    alpha = 0.0
    h_val = np.inf

    bounds = []

    # Positive weight bounds
    for i in range(d):
        for j in range(d):
            if i == j:
                bounds.append((0, 0))
            else:
                bounds.append((0, None))

    # Negative weight bounds
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
                options={"maxiter": 1000}
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

        print(f"NOTEARS iteration {iteration + 1}: h = {h_val:.10f}, rho = {rho:.1e}")

        if h_val <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est, d)
    W_est[np.abs(W_est) < w_threshold] = 0

    return W_est


# ============================================================
# DATA LOADING + PREPROCESSING
# ============================================================

def load_and_prepare_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Make sure job_table.parquet is inside data/hpc/"
        )

    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)

    print("Original dataset shape:", df.shape)

    numeric_df = df.select_dtypes(include=["number"]).copy()

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    # Drop columns with more than 25% missing values
    min_non_missing = int(0.75 * len(numeric_df))
    numeric_df = numeric_df.dropna(axis=1, thresh=min_non_missing)

    # Fill remaining missing values with median
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    # Remove constant columns
    nunique = numeric_df.nunique()
    numeric_df = numeric_df.loc[:, nunique > 1]

    # Select top numeric features by variance
    variances = numeric_df.var().sort_values(ascending=False)
    selected_cols = variances.head(12).index.tolist()

    print("\nSelected features for NOTEARS:")
    for col in selected_cols:
        print("-", col)

    df_selected = numeric_df[selected_cols].copy()

    # Sample for first single NOTEARS run to keep it fast
    if len(df_selected) > 5000:
        df_selected = df_selected.sample(n=5000, random_state=42)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_selected)

    return X, selected_cols


# ============================================================
# ATTRACTIVE DAG VISUALIZATION
# ============================================================

def draw_fancy_dag(W, labels, output_path, title, threshold=0.0):
    target_keywords = [
        "energy",
        "power",
        "time",
        "runtime",
        "duration",
        "elapsed",
        "wall",
        "cpu",
        "memory"
    ]

    G = nx.DiGraph()

    for label in labels:
        G.add_node(label)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            weight = W[i, j]
            if abs(weight) > threshold:
                G.add_edge(labels[i], labels[j], weight=weight)

    plt.figure(figsize=(16, 10))
    ax = plt.gca()
    ax.set_facecolor("#0b1020")

    if len(G.edges()) == 0:
        plt.text(
            0.5,
            0.5,
            "No edges found at this threshold",
            ha="center",
            va="center",
            fontsize=18,
            color="white"
        )
        plt.axis("off")
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#0b1020")
        plt.close()
        return

    pos = nx.spring_layout(G, seed=42, k=1.2)

    node_colors = []
    node_sizes = []

    for node in G.nodes():
        node_lower = node.lower()

        if any(key in node_lower for key in target_keywords):
            node_colors.append("#ff6b35")
            node_sizes.append(2800)
        else:
            node_colors.append("#4cc9f0")
            node_sizes.append(2000)

    edge_weights = [abs(G[u][v]["weight"]) for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1.0

    edge_widths = [
        1.5 + 5.0 * (abs(G[u][v]["weight"]) / max_weight)
        for u, v in G.edges()
    ]

    edge_colors = [
        "#ffb703" if G[u][v]["weight"] > 0 else "#c77dff"
        for u, v in G.edges()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white",
        linewidths=2.2,
        alpha=0.96
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=24,
        width=edge_widths,
        edge_color=edge_colors,
        connectionstyle="arc3,rad=0.13",
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
        bbox=dict(
            facecolor="#111827",
            edgecolor="none",
            alpha=0.75
        )
    )

    plt.title(
        title,
        fontsize=20,
        fontweight="bold",
        color="white",
        pad=20
    )

    legend_text = (
        "Orange nodes = performance / power / energy related features\n"
        "Yellow edges = positive NOTEARS weight | Purple edges = negative NOTEARS weight\n"
        "Thicker arrows = stronger learned causal dependency"
    )

    plt.text(
        0.01,
        0.01,
        legend_text,
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        bbox=dict(
            facecolor="#111827",
            edgecolor="#4cc9f0",
            alpha=0.85,
            boxstyle="round,pad=0.6"
        )
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#0b1020")
    plt.close()


def draw_causal_chain(W, labels, output_path):
    target_keywords = [
        "energy",
        "power",
        "time",
        "runtime",
        "duration",
        "elapsed",
        "wall"
    ]

    target_candidates = []

    for label in labels:
        lower = label.lower()
        if any(key in lower for key in target_keywords):
            target_candidates.append(label)

    if target_candidates:
        target = target_candidates[0]
    else:
        target = labels[-1]

    target_idx = labels.index(target)

    incoming = []

    for i in range(W.shape[0]):
        if abs(W[i, target_idx]) > 0:
            incoming.append((labels[i], target, W[i, target_idx]))

    incoming = sorted(incoming, key=lambda x: abs(x[2]), reverse=True)[:5]

    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    ax.set_facecolor("#101820")

    if not incoming:
        plt.text(
            0.5,
            0.5,
            f"No direct incoming causal chain found for {target}",
            ha="center",
            va="center",
            fontsize=16,
            color="white"
        )
        plt.axis("off")
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#101820")
        plt.close()
        return

    G = nx.DiGraph()

    for u, v, w in incoming:
        G.add_edge(u, v, weight=w)

    pos = {}

    left_nodes = [u for u, _, _ in incoming]

    for idx, node in enumerate(left_nodes):
        pos[node] = (0, idx)

    pos[target] = (2.2, len(left_nodes) / 2)

    node_colors = [
        "#ff6b35" if node == target else "#2ec4b6"
        for node in G.nodes()
    ]

    node_sizes = [
        3300 if node == target else 2300
        for node in G.nodes()
    ]

    max_w = max(abs(x[2]) for x in incoming)

    edge_widths = [
        2.0 + 6.0 * (abs(G[u][v]["weight"]) / max_w)
        for u, v in G.edges()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white",
        linewidths=2.3
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=26,
        width=edge_widths,
        edge_color="#ffd166",
        connectionstyle="arc3,rad=0.08"
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
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
        font_size=9,
        font_color="white",
        bbox=dict(
            facecolor="#111827",
            edgecolor="none",
            alpha=0.8
        )
    )

    plt.title(
        f"Top NOTEARS Causal Chain Toward {target}",
        fontsize=20,
        fontweight="bold",
        color="white",
        pad=20
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#101820")
    plt.close()


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_edges(W, labels, edge_path):
    with open(edge_path, "w", encoding="utf-8") as f:
        edge_count = 0

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if abs(W[i, j]) > 0:
                    edge_count += 1
                    f.write(
                        f"{labels[i]} -> {labels[j]} | weight = {W[i, j]:.4f}\n"
                    )

        if edge_count == 0:
            f.write("No edges found with current threshold.\n")


# ============================================================
# MAIN
# ============================================================

def main():
    X, labels = load_and_prepare_data()

    print("\nRunning NOTEARS causal discovery...")
    W = notears_linear(
        X,
        lambda1=0.02,
        max_iter=20,
        w_threshold=0.20
    )

    adjacency_path = os.path.join(GLOBAL_DIR, "pm100_notears_adjacency.csv")
    edge_path = os.path.join(GLOBAL_DIR, "pm100_notears_edges.txt")

    pd.DataFrame(W, index=labels, columns=labels).to_csv(adjacency_path)
    save_edges(W, labels, edge_path)

    print("\nGenerated edges:")
    with open(edge_path, "r", encoding="utf-8") as f:
        print(f.read())

    full_dag_path = os.path.join(PLOT_DIR, "pm100_notears_full_dag.png")
    filtered_dag_path = os.path.join(PLOT_DIR, "pm100_notears_filtered_dag.png")
    chain_path = os.path.join(PLOT_DIR, "pm100_notears_causal_chain.png")

    draw_fancy_dag(
        W,
        labels,
        full_dag_path,
        "PM100 NOTEARS Full Causal DAG",
        threshold=0.0
    )

    draw_fancy_dag(
        W,
        labels,
        filtered_dag_path,
        "PM100 NOTEARS Filtered Strong Causal DAG",
        threshold=0.35
    )

    draw_causal_chain(
        W,
        labels,
        chain_path
    )

    print("\nDONE.")
    print("Saved adjacency:", adjacency_path)
    print("Saved edge list:", edge_path)
    print("Saved plots:")
    print("-", full_dag_path)
    print("-", filtered_dag_path)
    print("-", chain_path)


if __name__ == "__main__":
    main()
