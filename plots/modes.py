# simple_plots.py
# Βάλε τα νούμερά σου στα lists και τρέξε:  python simple_plots.py
# Output: time_mean.png, coverage_mean.png

import numpy as np
import matplotlib.pyplot as plt

modes = ["availability_critical", "normal", "security_critical"]
inj_sizes = [1009, 3713, 7769]  # bytes

# === ΒΑΛΕ ΕΔΩ ΤΑ ΔΙΚΑ ΣΟΥ ΝΟΥΜΕΡΑ (5 τιμές για κάθε mode & size) ===
# times in seconds
times = {
    1009: {
        "availability_critical": [0, 0, 0, 0, 0],
        "normal":               [0, 0, 0, 0, 0],
        "security_critical":    [0, 0, 0, 0, 0],
    },
    3713: {
        "availability_critical": [0.1, 40.412, 50.563, 55.722, 5],
        "normal":               [0, 0, 0, 0, 0],
        "security_critical":    [0, 0, 0, 0, 0],
    },
    7769: {
        "availability_critical": [0, 0, 0, 0, 0],
        "normal":               [0, 0, 0, 0, 0],
        "security_critical":    [0, 0, 0, 0, 0],
    },
}

# coverage as "full passes" (FULL=1, PARTIAL=k/block_count, accumulate until detection)
coverage = {
    1009: {
        "availability_critical": [0, 0, 0, 0, 0],
        "normal":               [0, 0, 0, 0, 0],
        "security_critical":    [0, 0, 0, 0, 0],
    },
    3713: {
        "availability_critical": [0.351, 1.013, 0.818, 1.172, 0.646],
        "normal":               [0, 0, 0, 0, 0],
        "security_critical":    [0, 0, 0, 0, 0],
    },
    7769: {
        "availability_critical": [0, 0, 0, 0, 0],
        "normal":               [0, 0, 0, 0, 0],
        "security_critical":    [0, 0, 0, 0, 0],
    },
}

# ===== compute means =====
time_means = {s: [np.mean(times[s][m]) for m in modes] for s in inj_sizes}
cov_means  = {s: [np.mean(coverage[s][m]) for m in modes] for s in inj_sizes}

x = np.arange(len(modes))
w = 0.25

# ===== plot mean time =====
plt.figure()
for i, s in enumerate(inj_sizes):
    plt.bar(x + (i-1)*w, time_means[s], w, label=f"{s} bytes")
plt.xticks(x, modes, rotation=15)
plt.ylabel("Mean detection time (s)")
plt.title("Mean detection time (5 runs)")
plt.legend()
plt.tight_layout()
plt.savefig("time_mean.png", dpi=200)

# ===== plot mean coverage =====
plt.figure()
for i, s in enumerate(inj_sizes):
    plt.bar(x + (i-1)*w, cov_means[s], w, label=f"{s} bytes")
plt.xticks(x, modes, rotation=15)
plt.ylabel("Mean coverage until detection (full passes)")
plt.title("Mean coverage until detection (5 runs)")
plt.legend()
plt.tight_layout()
plt.savefig("coverage_mean.png", dpi=200)

print("Saved: time_mean.png, coverage_mean.png")
