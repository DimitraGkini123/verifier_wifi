import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Replace with YOUR measurements
# (time to first UNTRUSTED FULL)
# -----------------------------
deterministic_times = [
    33.8, 34.1, 33.9, 34.0, 34.2,
]

random_lru_times = [
    21.5, 41.6, 71.8, 21.8, 42.7,
]
det_median = np.median(deterministic_times)
rnd_median = np.median(random_lru_times)

det_q1, det_q3 = np.percentile(deterministic_times, [25, 75])
rnd_q1, rnd_q3 = np.percentile(random_lru_times, [25, 75])
labels = ["Deterministic LRU", "Randomized LRU"]
medians = [det_median, rnd_median]

plt.figure(figsize=(6.2, 4.0))
bars = plt.bar(labels, medians)

plt.ylabel("Median detection time (s)")
plt.title("Median attack detection time (192 memory partitions)")
plt.grid(axis="y", alpha=0.3)

# annotate bars
for bar, val in zip(bars, medians):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{val:.2f}s",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.show()
iqr_low  = [det_median - det_q1, rnd_median - rnd_q1]
iqr_high = [det_q3 - det_median, rnd_q3 - rnd_median]

plt.figure(figsize=(6.2, 4.0))
plt.bar(labels, medians, yerr=[iqr_low, iqr_high], capsize=6)

plt.ylabel("Median detection time (s)")
plt.title("Median ± IQR detection time (192 memory partitions)")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
