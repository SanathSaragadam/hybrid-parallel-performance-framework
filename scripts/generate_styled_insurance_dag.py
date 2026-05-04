import os
import re

EDGE_FILE = "results/global/insurance_global_dag_edges.txt"
DOT_FILE = "results/global/insurance_global_dag.dot"

os.makedirs("results/global", exist_ok=True)

edges = []

if os.path.exists(EDGE_FILE):
    with open(EDGE_FILE, "r") as f:
        for line in f:
            line = line.strip()
            match = re.match(r"(.+?)\s*->\s*(.+?)\s+freq\s*=\s*(\d+)", line)
            if match:
                left = match.group(1).strip()
                right = match.group(2).strip()
                freq = int(match.group(3))
                edges.append((left, right, freq))

with open(DOT_FILE, "w") as f:
    f.write("digraph Insurance_GlobalDAG {\n")
    f.write("    rankdir=LR;\n")
    f.write("    bgcolor=\"white\";\n")
    f.write("    node [shape=ellipse, style=filled, fillcolor=lightyellow, fontname=\"Helvetica\", fontsize=11];\n")
    f.write("    edge [color=gray40, arrowsize=0.9];\n")
    f.write("    \"charges\" [fillcolor=lightcoral, shape=doublecircle];\n")

    for left, right, freq in edges:
        penwidth = 1.0 + (0.6 * freq)
        color = "gray40"
        if right == "charges":
            color = "firebrick"
        f.write(f'    "{left}" -> "{right}" [penwidth={penwidth}, color="{color}"];\n')

    f.write("}\n")

print(f"Styled DAG DOT file generated: {DOT_FILE}")
