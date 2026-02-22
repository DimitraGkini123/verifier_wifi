## detection time and memory coverage for attack detection under different modes
## injectiom of 65065 bytes

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# EXPERIMENTAL RUNS
# ==========================================

# Detection times (seconds)
baseline_times     = [0.2, 0.15]   # Full hash every 5s
availability_times = [33.52, 40.2]
normal_times       = [18, 30.1]
security_times     = [9, 6]

# Memory traversals (1.0 = one full firmware traversal)
baseline_mem       = [1.0, 1.0]
availability_mem   = [0.36, 0.36]
normal_mem         = [0.5, 0.7]
security_mem       = [0.75, 0.6]

# ==========================================
# MEANS
# ==========================================

modes = ["Full (5s baseline)", "Availability", "Normal", "Security"]
colors = ["gray", "green", "orange", "red"]

time_data = [baseline_times, availability_times, normal_times, security_times]
mem_data  = [baseline_mem, availability_mem, normal_mem, security_mem]

time_means = [np.mean(x) for x in time_data]
mem_means  = [np.mean(x) for x in mem_data]

# ==========================================
# PLOT 1: Detection Time
# ==========================================

plt.figure()
plt.bar(modes, time_means, color=colors)
plt.ylabel("Detection Time (s)")
plt.title("Detection Time vs Mode (mean of runs)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# ==========================================
# PLOT 2: Memory Traversed
# ==========================================

plt.figure()
plt.bar(modes, mem_means, color=colors)
plt.ylabel("Firmware Traversals (1.0 = Full Hash)")
plt.title("Memory Traversed Until Detection (mean of runs)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()