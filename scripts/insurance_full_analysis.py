import os
import pandas as pd
import matplotlib.pyplot as plt

DATASET = "data/main/insurance.csv"
PLOTS_DIR = "results/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(DATASET)

# Encode categorical
df["sex"] = df["sex"].astype("category").cat.codes
df["smoker"] = df["smoker"].astype("category").cat.codes
df["region"] = df["region"].astype("category").cat.codes

# -----------------------------
# 1. Correlation Heatmap
# -----------------------------
corr = df.corr()

plt.figure(figsize=(8,6))
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Insurance Correlation Heatmap")
plt.tight_layout()
plt.savefig(PLOTS_DIR + "/full_correlation_heatmap.png")
plt.close()

# -----------------------------
# 2. Charges Influence
# -----------------------------
charges_corr = corr["charges"].drop("charges")

plt.figure()
charges_corr.plot(kind='bar')
plt.title("Feature Influence on Charges")
plt.tight_layout()
plt.savefig(PLOTS_DIR + "/charges_influence.png")
plt.close()

# -----------------------------
# 3. Smoker vs Charges
# -----------------------------
plt.figure()
df.boxplot(column="charges", by="smoker")
plt.title("Smoker vs Charges")
plt.suptitle("")
plt.savefig(PLOTS_DIR + "/smoker_vs_charges.png")
plt.close()

# -----------------------------
# 4. BMI vs Charges
# -----------------------------
plt.figure()
plt.scatter(df["bmi"], df["charges"])
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.title("BMI vs Charges")
plt.savefig(PLOTS_DIR + "/bmi_vs_charges.png")
plt.close()

# -----------------------------
# 5. Conditional (Smoker Split)
# -----------------------------
plt.figure()

smoker_0 = df[df["smoker"] == 0]
smoker_1 = df[df["smoker"] == 1]

plt.scatter(smoker_0["bmi"], smoker_0["charges"], label="Non-Smoker")
plt.scatter(smoker_1["bmi"], smoker_1["charges"], label="Smoker")

plt.legend()
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.title("BMI vs Charges conditioned on Smoker")
plt.savefig(PLOTS_DIR + "/conditional_plot.png")
plt.close()

print("Full insurance analysis graphs generated!")
