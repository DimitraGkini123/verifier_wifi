import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Fill these with YOUR numbers
# -----------------------------
# Time to detection (seconds) in availability-critical policy
t_detect_256 = 245.9  # <-- replace with your measured time when FW_BLOCKS_N=256
t_detect_128 = 63  # <-- replace with your measured time when FW_BLOCKS_N=128

labels = ["256 blocks", "128 blocks"]
times  = [t_detect_256, t_detect_128]

# -----------------------------
# Plot
# -----------------------------
x = np.arange(len(labels))

plt.figure(figsize=(6.2, 4.0))
bars = plt.bar(x, times)

plt.xticks(x, labels)
plt.ylabel("Time to attack detection (s)")
plt.title("Availability-critical: detection time vs memory partitioning")
plt.grid(True, axis="y", alpha=0.3)

# annotate bars with time
for b, t in zip(bars, times):
    plt.text(
        b.get_x() + b.get_width() / 2,
        b.get_height(),
        f"{t:.2f}s",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.show()
