import matplotlib.pyplot as plt
import pandas as pd

# 調整圖的長寬比例 (units: inches)
fig, ax = plt.subplots(figsize=(10, 5))
# Prepare data
data = {
    "size": [748, 2862, 557, 909, 8290, 809, 3852, 827, 324, 3136],
    "old": [0.00, 35.82, 54.69, 65.77, 94.38, 51.13, 96.67, 0.00, 2.35, 48.40],
    "Normalize": [11.29, 12.09, 21.66, 52.58, 93.86, 14.66, 39.24, 13.50, 19.42, 8.84],
    "Normalize & Weight": [
        11.90,
        45.40,
        0.00,
        0.00,
        94.62,
        0.00,
        92.26,
        0.00,
        19.28,
        84.88,
    ],
    "weight": [
        "0.00%",
        "0.213115",
        "0.226667",
        "0.275449",
        "0.949921",
        "0.461538",
        "0.95721",
        "0.00%",
        "0.00%",
        "0.203488",
    ],
}
df = pd.DataFrame(data)

# Convert percentage columns to decimal fractions
for col in ["old", "Normalize", "Normalize & Weight"]:
    df[col] = df[col] / 100.0


# Parse weight column: percentages vs decimals
def parse_weight(x):
    if isinstance(x, str) and x.endswith("%"):
        return float(x.rstrip("%")) / 100.0
    else:
        return float(x)


df["weight"] = df["weight"].apply(parse_weight)

# New Tailwind palette
palette = {
    "old": "#1c1018",  # Licorice
    "Normalize": "#2d82b7",  # Steel blue
    "Normalize & Weight": "#42e2b8",  # Aquamarine
    "weight": "#826754",  # Coyote
}

# Create scatter plot with specified colors, no grid
fig, ax = plt.subplots()

markers = {"old": "o", "Normalize": "s", "Normalize & Weight": "^", "weight": "x"}

for col, marker in markers.items():
    ax.scatter(df["size"], df[col], marker=marker, color=palette[col], label=col)

# Remove grid
ax.grid(False)

# Add guideline lines
ax.axvline(3000, linestyle="--", linewidth=0.5)
ax.axhline(0.65, linestyle="--", linewidth=0.5)

# Labels and legend
ax.set_xlabel("Model Size")
ax.set_ylabel("F1 Score")
ax.legend(loc="best")

# Save as PDF
output_path = "./svm_f1_vs_size_palette2.pdf"
fig.savefig(output_path, format="pdf", bbox_inches="tight")

output_path
