import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# Put *all runs* here (seconds) for each setting
# -------------------------------------------------
times_256 = [245.9, 250.1, 238.7, 261.3, 247.4]  # <-- replace with your runs
times_128 = [63.0, 58.4, 66.2, 61.9, 64.7]       # <-- replace with your runs

labels = ["256 blocks", "128 blocks"]
groups = [times_256, times_128]

# -------------------------------------------------
# Compute median + IQR (Q1, Q3)
# -------------------------------------------------
medians = [np.median(g) for g in groups]
q1s     = [np.percentile(g, 25) for g in groups]
q3s     = [np.percentile(g, 75) for g in groups]
iqrs    = [q3 - q1 for q1, q3 in zip(q1s, q3s)]

# For error bars around the median: distances to Q1 and Q3
yerr = np.array([
    [m - q1 for m, q1 in zip(medians, q1s)],   # lower
    [q3 - m for m, q3 in zip(medians, q3s)],   # upper
])

# -------------------------------------------------
# Plot: bars = median, error bars = IQR (Q1..Q3)
# -------------------------------------------------
x = np.arange(len(labels))

plt.figure(figsize=(6.6, 4.2))
bars = plt.bar(x, medians, yerr=yerr, capsize=6)

plt.xticks(x, labels)
plt.ylabel("Time to attack detection (s)")
plt.title("Availability-critical: detection time (median + IQR)")
plt.grid(True, axis="y", alpha=0.3)

# annotate: median + IQR
for i, b in enumerate(bars):
    m = medians[i]
    q1, q3 = q1s[i], q3s[i]
    plt.text(
        b.get_x() + b.get_width() / 2,
        b.get_height(),
        f"median={m:.2f}s\nIQR=[{q1:.2f}, {q3:.2f}]",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.show()

# -------------------------------------------------
# Also print stats neatly (useful for thesis tables)
# -------------------------------------------------
for name, g, m, q1, q3, iqr in zip(labels, groups, medians, q1s, q3s, iqrs):
    print(f"{name}: n={len(g)}, median={m:.2f}s, Q1={q1:.2f}s, Q3={q3:.2f}s, IQR={iqr:.2f}s")
