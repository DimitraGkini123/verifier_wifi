import matplotlib.pyplot as plt
import numpy as np

labels = [
    "FULL every 5s (baseline)",
    "Policy: availability-critical",
    "Policy: balanced",
    "Policy: security-first",
]

# Coverage until first detection in "full-image equivalents"
# (FULL=1.0, PARTIAL=k/FW_BLOCKS_N, sum until first alarm)
coverage_traversals = [
    5.0,    # <-- replace
    3.03,   # <-- replace
    3.41,   # <-- replace
    4.14,   # <-- replace
]

# Time-to-detection in seconds
time_to_detection_s = [
    15.632,  # <-- replace
    245.9,   # <-- replace
    74.3,    # <-- replace
    72.0,    # <-- replace
]

x = np.arange(len(labels))

# -----------------------------
# Plot 1: Memory coverage
# -----------------------------
plt.figure(figsize=(9, 4.6))
bars = plt.bar(x, coverage_traversals)
plt.xticks(x, labels, rotation=0, ha="center")
plt.ylabel("Memory coverage until detection\n(full-image equivalents)")
plt.title("Memory coverage until attack detection")
plt.grid(True, axis="y", alpha=0.3)

for i, b in enumerate(bars):
    v = coverage_traversals[i]
    plt.text(
        b.get_x() + b.get_width() / 2,
        b.get_height(),
        f"{v:.3f}×",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.show()

# -----------------------------
# Plot 2: Time to detection
# -----------------------------
plt.figure(figsize=(9, 4.6))
bars = plt.bar(x, time_to_detection_s)
plt.xticks(x, labels, rotation=0, ha="center")
plt.ylabel("Time to detection (s)")
plt.title("Time to attack detection")
plt.grid(True, axis="y", alpha=0.3)

for i, b in enumerate(bars):
    t = time_to_detection_s[i]
    plt.text(
        b.get_x() + b.get_width() / 2,
        b.get_height(),
        f"{t:.2f}s",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.show()
