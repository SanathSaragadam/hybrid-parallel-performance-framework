import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/hpc/job_table.parquet"

GLOBAL_EDGE_FILE = "results/notears/global/pm100_global_stable_edges.csv"
GLOBAL_ADJ_FILE = "results/notears/global/pm100_global_stable_adjacency.csv"

PLOT_DIR = "results/notears/plots"
METRIC_DIR = "results/notears/metrics"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)


def load_selected_dataset():
    df = pd.read_parquet(DATA_PATH)
    numeric_df = df.select_dtypes(include=["number"]).copy()

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.dropna(axis=1, thresh=int(0.75 * len(numeric_df)))
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    nunique = numeric_df.nunique()
    numeric_df = numeric_df.loc[:, nunique > 1]

    variances = numeric_df.var().sort_values(ascending=False)
    selected_cols = variances.head(12).index.tolist()

    selected_df = numeric_df[selected_cols].copy()

    if len(selected_df) > 10000:
        selected_df = selected_df.sample(n=10000, random_state=42)

    return selected_df, selected_cols


def choose_target(columns):
    keywords = ["energy", "power", "runtime", "duration", "elapsed", "time", "wall"]

    for key in keywords:
        for col in columns:
            if key in col.lower():
                return col

    return columns[-1]


def draw_correlation_heatmap(df, output_path):
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(13, 10))
    ax = plt.gca()
    ax.set_facecolor("#0b1020")

    im = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8, color="white")
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8, color="white")

    plt.title(
        "PM100 Feature Relationship Heatmap",
        fontsize=18,
        fontweight="bold",
        color="white",
        pad=18
    )

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            value = corr.values[i, j]
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white")

    for spine in ax.spines.values():
        spine.set_color("white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#0b1020")
    plt.close()


def draw_correlation_vs_notears(df, edge_df, target, output_path):
    correlations = df.corr(numeric_only=True)[target].drop(target).abs().sort_values(ascending=False)

    corr_df = pd.DataFrame({
        "feature": correlations.index,
        "correlation_strength": correlations.values
    })

    causal_df = edge_df[edge_df["target"] == target].copy()

    if causal_df.empty:
        causal_df = edge_df.head(10).copy()
        causal_df["feature"] = causal_df["source"]
    else:
        causal_df["feature"] = causal_df["source"]

    causal_strength = causal_df.groupby("feature")["avg_abs_weight"].max().reset_index()
    causal_strength = causal_strength.rename(columns={"avg_abs_weight": "notears_strength"})

    compare = corr_df.merge(causal_strength, on="feature", how="outer").fillna(0)
    compare = compare.sort_values(
        by=["notears_strength", "correlation_strength"],
        ascending=False
    ).head(12)

    x = np.arange(len(compare))
    width = 0.38

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    ax.set_facecolor("#101820")

    plt.bar(x - width / 2, compare["correlation_strength"], width, label="Correlation Strength")
    plt.bar(x + width / 2, compare["notears_strength"], width, label="NOTEARS Causal Strength")

    plt.xticks(x, compare["feature"], rotation=45, ha="right", fontsize=8, color="white")
    plt.yticks(color="white")

    plt.ylabel("Normalized / Absolute Strength", color="white", fontsize=12)
    plt.title(
        f"Correlation vs NOTEARS Causal Influence Toward {target}",
        fontsize=18,
        fontweight="bold",
        color="white",
        pad=18
    )

    legend = plt.legend()
    legend.get_frame().set_alpha(0.85)

    for spine in ax.spines.values():
        spine.set_color("white")

    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#101820")
    plt.close()


def draw_stability_filtering_story(edge_df, output_path):
    if edge_df.empty:
        return

    bins = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ["Weak", "Moderate", "Strong", "Very Stable"]

    temp = edge_df.copy()
    temp["stability_group"] = pd.cut(
        temp["stability"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    counts = temp["stability_group"].value_counts().reindex(labels).fillna(0)

    plt.figure(figsize=(11, 7))
    ax = plt.gca()
    ax.set_facecolor("#0b1020")

    plt.bar(counts.index.astype(str), counts.values)

    plt.title(
        "Stable Edge Filtering Analysis",
        fontsize=18,
        fontweight="bold",
        color="white",
        pad=18
    )

    plt.xlabel("Edge Stability Category", color="white", fontsize=12)
    plt.ylabel("Number of Retained Global Edges", color="white", fontsize=12)

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    for i, value in enumerate(counts.values):
        plt.text(i, value + 0.2, str(int(value)), ha="center", color="white", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#0b1020")
    plt.close()


def draw_causal_chain_from_global(edge_df, target, output_path):
    target_edges = edge_df[edge_df["target"] == target].copy()

    if target_edges.empty:
        target_edges = edge_df.sort_values("avg_abs_weight", ascending=False).head(6).copy()

    target_edges = target_edges.sort_values("avg_abs_weight", ascending=False).head(6)

    G = nx.DiGraph()

    for _, row in target_edges.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            weight=row["avg_abs_weight"],
            stability=row["stability"]
        )

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    ax.set_facecolor("#060b16")

    if len(G.edges()) == 0:
        plt.text(
            0.5,
            0.5,
            "No causal chain edges available",
            ha="center",
            va="center",
            color="white",
            fontsize=18
        )
        plt.axis("off")
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#060b16")
        plt.close()
        return

    left_nodes = list(set(target_edges["source"].tolist()))
    target_node = target_edges.iloc[0]["target"]

    pos = {}
    for i, node in enumerate(left_nodes):
        pos[node] = (0, i)

    pos[target_node] = (2.4, len(left_nodes) / 2)

    node_colors = [
        "#ff4d00" if node == target_node else "#2ec4b6"
        for node in G.nodes()
    ]

    node_sizes = [
        3600 if node == target_node else 2400
        for node in G.nodes()
    ]

    max_w = max([G[u][v]["weight"] for u, v in G.edges()])
    edge_widths = [
        2 + 7 * (G[u][v]["weight"] / max_w)
        for u, v in G.edges()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white",
        linewidths=2.5
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=30,
        width=edge_widths,
        edge_color="#ffd60a",
        connectionstyle="arc3,rad=0.1"
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_weight="bold",
        font_color="black"
    )

    edge_labels = {
        (u, v): f"W={G[u][v]['weight']:.2f}\nS={G[u][v]['stability']:.2f}"
        for u, v in G.edges()
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="white",
        bbox=dict(facecolor="#111827", edgecolor="none", alpha=0.8)
    )

    plt.title(
        f"Stable Causal Chain Toward {target_node}",
        fontsize=20,
        fontweight="bold",
        color="white",
        pad=20
    )

    plt.text(
        0.01,
        0.01,
        "W = average NOTEARS weight | S = stability across chunks",
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        bbox=dict(facecolor="#111827", edgecolor="#2ec4b6", alpha=0.85, boxstyle="round,pad=0.6")
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#060b16")
    plt.close()


def write_final_summary(df, edge_df, target):
    summary_path = os.path.join(METRIC_DIR, "pm100_final_analysis_pack_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PM100 NOTEARS Final Analysis Pack Summary\n")
        f.write("========================================\n\n")

        f.write(f"Rows used for analysis view: {len(df)}\n")
        f.write(f"Selected features: {list(df.columns)}\n")
        f.write(f"Primary target used for interpretation: {target}\n\n")

        f.write("Generated analysis outputs:\n")
        f.write("- Feature relationship heatmap\n")
        f.write("- Correlation vs NOTEARS comparison\n")
        f.write("- Stable edge filtering analysis\n")
        f.write("- Global causal chain visualization\n\n")

        f.write("Interpretation:\n")
        f.write(
            "The heatmap provides a statistical view of feature relationships, while NOTEARS "
            "learns a directed acyclic graph. The comparison plot helps separate simple "
            "correlation from learned directed causal dependencies. The causal chain plot "
            "summarizes the strongest stable incoming dependencies toward the selected "
            "performance or energy-related target.\n\n"
        )

        if not edge_df.empty:
            f.write("Top global stable edges:\n")
            for _, row in edge_df.head(10).iterrows():
                f.write(
                    f"- {row['source']} -> {row['target']} "
                    f"(weight={row['avg_abs_weight']:.4f}, stability={row['stability']:.2f})\n"
                )

    print("Saved final analysis summary:", summary_path)


def main():
    df, selected_cols = load_selected_dataset()

    if not os.path.exists(GLOBAL_EDGE_FILE):
        raise FileNotFoundError("Global stable edge file not found. Run merge_notears_dags.py first.")

    edge_df = pd.read_csv(GLOBAL_EDGE_FILE)

    target = choose_target(selected_cols)

    print("Selected interpretation target:", target)

    draw_correlation_heatmap(
        df,
        os.path.join(PLOT_DIR, "pm100_feature_relationship_heatmap.png")
    )

    draw_correlation_vs_notears(
        df,
        edge_df,
        target,
        os.path.join(PLOT_DIR, "pm100_correlation_vs_notears_influence.png")
    )

    draw_stability_filtering_story(
        edge_df,
        os.path.join(PLOT_DIR, "pm100_stability_filtering_analysis.png")
    )

    draw_causal_chain_from_global(
        edge_df,
        target,
        os.path.join(PLOT_DIR, "pm100_global_causal_chain_analysis.png")
    )

    write_final_summary(df, edge_df, target)

    print("\nDONE.")
    print("Generated final analysis plots:")
    print("- results/notears/plots/pm100_feature_relationship_heatmap.png")
    print("- results/notears/plots/pm100_correlation_vs_notears_influence.png")
    print("- results/notears/plots/pm100_stability_filtering_analysis.png")
    print("- results/notears/plots/pm100_global_causal_chain_analysis.png")


if __name__ == "__main__":
    main()
