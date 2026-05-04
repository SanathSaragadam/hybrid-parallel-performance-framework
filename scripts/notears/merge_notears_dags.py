import os
import re
import glob
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


LOCAL_DIR = "results/notears/local_dags"
GLOBAL_DIR = "results/notears/global"
PLOT_DIR = "results/notears/plots"
METRIC_DIR = "results/notears/metrics"

os.makedirs(GLOBAL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)


def load_chunk_adjacencies():
    files = sorted(glob.glob(os.path.join(LOCAL_DIR, "pm100_chunk_*_adjacency.csv")))

    if not files:
        raise FileNotFoundError(
            "No chunk adjacency files found. Run run_notears_chunks.py first."
        )

    matrices = []
    labels = None

    for file in files:
        df = pd.read_csv(file, index_col=0)

        if labels is None:
            labels = df.columns.tolist()

        matrices.append(df.values)

    return matrices, labels, files


def merge_adjacencies(matrices):
    stack = np.stack(matrices, axis=0)

    frequency = np.sum(np.abs(stack) > 0, axis=0)
    avg_weight = np.mean(stack, axis=0)
    avg_abs_weight = np.mean(np.abs(stack), axis=0)

    num_chunks = len(matrices)
    stability = frequency / num_chunks

    global_W = np.zeros_like(avg_weight)

    for i in range(global_W.shape[0]):
        for j in range(global_W.shape[1]):
            if i == j:
                continue

            if stability[i, j] >= 0.40 and avg_abs_weight[i, j] >= 0.10:
                global_W[i, j] = avg_weight[i, j]

    return global_W, frequency, stability, avg_weight, avg_abs_weight


def save_global_outputs(global_W, frequency, stability, avg_weight, avg_abs_weight, labels):
    pd.DataFrame(global_W, index=labels, columns=labels).to_csv(
        os.path.join(GLOBAL_DIR, "pm100_global_stable_adjacency.csv")
    )

    pd.DataFrame(frequency, index=labels, columns=labels).to_csv(
        os.path.join(GLOBAL_DIR, "pm100_edge_frequency_matrix.csv")
    )

    pd.DataFrame(stability, index=labels, columns=labels).to_csv(
        os.path.join(GLOBAL_DIR, "pm100_edge_stability_matrix.csv")
    )

    edge_rows = []

    for i in range(global_W.shape[0]):
        for j in range(global_W.shape[1]):
            if abs(global_W[i, j]) > 0:
                edge_rows.append({
                    "source": labels[i],
                    "target": labels[j],
                    "avg_weight": avg_weight[i, j],
                    "avg_abs_weight": avg_abs_weight[i, j],
                    "frequency": int(frequency[i, j]),
                    "stability": stability[i, j]
                })

    edge_df = pd.DataFrame(edge_rows)
    edge_df = edge_df.sort_values(
        by=["stability", "avg_abs_weight"],
        ascending=False
    )

    edge_df.to_csv(
        os.path.join(GLOBAL_DIR, "pm100_global_stable_edges.csv"),
        index=False
    )

    with open(os.path.join(GLOBAL_DIR, "pm100_global_stable_edges.txt"), "w", encoding="utf-8") as f:
        if edge_df.empty:
            f.write("No stable edges found with current merge threshold.\n")
        else:
            for _, row in edge_df.iterrows():
                f.write(
                    f"{row['source']} -> {row['target']} | "
                    f"avg_weight={row['avg_weight']:.4f} | "
                    f"frequency={int(row['frequency'])} | "
                    f"stability={row['stability']:.2f}\n"
                )

    return edge_df


def draw_global_dag(global_W, labels, edge_df):
    G = nx.DiGraph()

    for label in labels:
        G.add_node(label)

    for _, row in edge_df.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            weight=row["avg_weight"],
            stability=row["stability"],
            frequency=row["frequency"]
        )

    plt.figure(figsize=(18, 11))
    ax = plt.gca()
    ax.set_facecolor("#070b1a")

    if len(G.edges()) == 0:
        plt.text(
            0.5,
            0.5,
            "No stable global edges found",
            ha="center",
            va="center",
            fontsize=20,
            color="white"
        )
        plt.axis("off")
        plt.savefig(
            os.path.join(PLOT_DIR, "pm100_global_stable_dag.png"),
            dpi=300,
            bbox_inches="tight",
            facecolor="#070b1a"
        )
        plt.close()
        return

    pos = nx.spring_layout(G, seed=7, k=1.6)

    target_keywords = ["energy", "power", "time", "runtime", "duration", "elapsed", "wall"]

    node_colors = []
    node_sizes = []

    for node in G.nodes():
        lower = node.lower()
        if any(k in lower for k in target_keywords):
            node_colors.append("#ff4d00")
            node_sizes.append(3300)
        else:
            node_colors.append("#00c2ff")
            node_sizes.append(2300)

    max_stability = max([G[u][v]["stability"] for u, v in G.edges()])
    edge_widths = [
        2.0 + 6.0 * (G[u][v]["stability"] / max_stability)
        for u, v in G.edges()
    ]

    edge_colors = [
        "#ffd60a" if G[u][v]["weight"] > 0 else "#c77dff"
        for u, v in G.edges()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white",
        linewidths=2.4,
        alpha=0.96
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=28,
        width=edge_widths,
        edge_color=edge_colors,
        connectionstyle="arc3,rad=0.14",
        alpha=0.9
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
        font_weight="bold",
        font_color="black"
    )

    edge_labels = {
        (u, v): f"S={G[u][v]['stability']:.2f}"
        for u, v in G.edges()
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="white",
        bbox=dict(facecolor="#111827", edgecolor="none", alpha=0.78)
    )

    plt.title(
        "PM100 Global Stable NOTEARS DAG",
        fontsize=22,
        fontweight="bold",
        color="white",
        pad=22
    )

    plt.text(
        0.01,
        0.01,
        "Edge label S = stability across local chunk DAGs\n"
        "Thicker arrows = more stable causal relationship\n"
        "Orange nodes = runtime / power / energy related variables",
        transform=ax.transAxes,
        fontsize=11,
        color="white",
        bbox=dict(facecolor="#111827", edgecolor="#00c2ff", alpha=0.88, boxstyle="round,pad=0.7")
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "pm100_global_stable_dag.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="#070b1a"
    )
    plt.close()


def draw_edge_stability_chart(edge_df):
    if edge_df.empty:
        return

    top_edges = edge_df.head(15).copy()
    top_edges["edge"] = top_edges["source"] + " → " + top_edges["target"]

    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    ax.set_facecolor("#0b1020")

    plt.barh(top_edges["edge"][::-1], top_edges["stability"][::-1])

    plt.xlabel("Stability Score Across Chunks", fontsize=12, color="white")
    plt.ylabel("Causal Edge", fontsize=12, color="white")
    plt.title(
        "Top Stable NOTEARS Edges Across Local DAGs",
        fontsize=18,
        fontweight="bold",
        color="white",
        pad=18
    )

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "pm100_edge_stability_chart.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="#0b1020"
    )
    plt.close()


def draw_local_global_comparison(matrices, labels, global_W):
    local_union = np.sum(np.abs(np.stack(matrices, axis=0)) > 0, axis=0)
    local_edge_count = int(np.sum(local_union > 0))
    global_edge_count = int(np.sum(np.abs(global_W) > 0))

    dropped_edges = local_edge_count - global_edge_count

    names = ["Local Union Edges", "Stable Global Edges", "Filtered Weak/Unstable Edges"]
    values = [local_edge_count, global_edge_count, max(dropped_edges, 0)]

    plt.figure(figsize=(11, 7))
    ax = plt.gca()
    ax.set_facecolor("#101820")

    plt.bar(names, values)

    plt.title(
        "Local DAGs vs Stable Global DAG Comparison",
        fontsize=18,
        fontweight="bold",
        color="white",
        pad=18
    )

    plt.ylabel("Number of Edges", fontsize=12, color="white")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("white")

    for i, value in enumerate(values):
        plt.text(i, value + 0.5, str(value), ha="center", color="white", fontsize=12)

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "pm100_local_vs_global_edge_comparison.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="#101820"
    )
    plt.close()


def draw_target_influence(edge_df):
    if edge_df.empty:
        return

    target_keywords = ["energy", "power", "time", "runtime", "duration", "elapsed", "wall"]

    target_edges = edge_df[
        edge_df["target"].str.lower().apply(
            lambda x: any(k in x for k in target_keywords)
        )
    ].copy()

    if target_edges.empty:
        target_edges = edge_df.head(10).copy()

    target_edges["edge"] = target_edges["source"] + " → " + target_edges["target"]
    target_edges = target_edges.sort_values("avg_abs_weight", ascending=False).head(12)

    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    ax.set_facecolor("#101820")

    plt.barh(target_edges["edge"][::-1], target_edges["avg_abs_weight"][::-1])

    plt.xlabel("Average Absolute NOTEARS Weight", fontsize=12, color="white")
    plt.ylabel("Influence Edge", fontsize=12, color="white")

    plt.title(
        "Strongest Causal Influences Toward Performance/Energy Variables",
        fontsize=18,
        fontweight="bold",
        color="white",
        pad=18
    )

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "pm100_target_causal_influence.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="#101820"
    )
    plt.close()


def write_analysis_summary(edge_df, matrices, global_W):
    summary_path = os.path.join(METRIC_DIR, "pm100_global_dag_analysis_summary.txt")

    local_union = np.sum(np.abs(np.stack(matrices, axis=0)) > 0, axis=0)
    local_edge_count = int(np.sum(local_union > 0))
    global_edge_count = int(np.sum(np.abs(global_W) > 0))

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PM100 NOTEARS Global DAG Analysis Summary\n")
        f.write("========================================\n\n")
        f.write(f"Number of local chunk DAGs analyzed: {len(matrices)}\n")
        f.write(f"Unique edges appearing in local DAGs: {local_edge_count}\n")
        f.write(f"Stable edges retained in global DAG: {global_edge_count}\n")
        f.write(f"Weak/unstable edges filtered out: {max(local_edge_count - global_edge_count, 0)}\n\n")

        f.write("Interpretation:\n")
        f.write(
            "Local DAGs capture causal dependencies within individual data partitions. "
            "The global DAG retains only edges that appear consistently across chunks, "
            "making the final causal structure more stable and reliable.\n\n"
        )

        f.write("Top stable edges:\n")
        if edge_df.empty:
            f.write("No stable edges found.\n")
        else:
            for _, row in edge_df.head(10).iterrows():
                f.write(
                    f"- {row['source']} -> {row['target']} "
                    f"(stability={row['stability']:.2f}, avg_weight={row['avg_weight']:.4f})\n"
                )

    print("Saved analysis summary:", summary_path)


def main():
    matrices, labels, files = load_chunk_adjacencies()

    print(f"Loaded {len(files)} local adjacency matrices.")

    global_W, frequency, stability, avg_weight, avg_abs_weight = merge_adjacencies(matrices)

    edge_df = save_global_outputs(
        global_W,
        frequency,
        stability,
        avg_weight,
        avg_abs_weight,
        labels
    )

    print("\nStable global edges:")
    if edge_df.empty:
        print("No stable edges found.")
    else:
        print(edge_df.head(15).to_string(index=False))

    draw_global_dag(global_W, labels, edge_df)
    draw_edge_stability_chart(edge_df)
    draw_local_global_comparison(matrices, labels, global_W)
    draw_target_influence(edge_df)
    write_analysis_summary(edge_df, matrices, global_W)

    print("\nDONE.")
    print("Generated:")
    print("- results/notears/plots/pm100_global_stable_dag.png")
    print("- results/notears/plots/pm100_edge_stability_chart.png")
    print("- results/notears/plots/pm100_local_vs_global_edge_comparison.png")
    print("- results/notears/plots/pm100_target_causal_influence.png")


if __name__ == "__main__":
    main()
