import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# ΒΑΛΕ ΕΔΩ ΤΑ ΝΟΥΜΕΡΑ ΣΟΥ (εσύ τα συμπληρώνεις)
# ---------------------------------------------------------
# Ιδέα: για κάθε mode δίνεις μια σειρά από "attestation events".
# Σε κάθε event:
#   - time_s: πότε έγινε (σε sec από την αρχή του run)
#   - blocks_hashed: πόσα blocks έγιναν hash σε αυτό το event (π.χ. k)
# Και μετά κάνουμε cumulative sum -> "cumulative blocks hashed".
#
# detection_time_s: ο χρόνος που έγινε detection (κατακόρυφη διακεκομμένη)
# detection_blocks: το cumulative blocks hashed τη στιγμή του detection
#                  (αν δεν το ξέρεις ακριβώς, βάλε π.χ. την τιμή που αντιστοιχεί
#                   στο τελευταίο event πριν/στο detection)
# =========================================================

DATA = {
    "Availability": {
        "color": "green",
        "time_s":  [0.0, 63.553, 64.955, 66.489],   # π.χ. [0, 30, 60, 90]
        "blocks_hashed":[39, 52, 256, 256],   # π.χ. [39, 52, 52, 52]
        "detection_time_s": 63.553,  # π.χ. 95.3
        "detection_blocks": 91,  # π.χ. 39+52+52
    },
    "Normal": {
        "color": "orange",
        "time_s":[0.0, 18.152, 19.607, 21.139],
        "blocks_hashed": [52, 77, 256, 256],
        "detection_time_s": 18.152,
        "detection_blocks": 129,
    },
    "Security": {
        "color": "red",
        "time_s":   [0.0, 19.329, 20.749, 22.388],
        "blocks_hashed": [77, 116, 256, 256],
        "detection_time_s": 19.329,
        "detection_blocks": 193,
    },
}

# =========================================================
# PLOT: cumulative blocks hashed vs time, + detection line
# =========================================================

plt.figure()

for mode, d in DATA.items():
    t = np.array(d["time_s"], dtype=float)
    b = np.array(d["blocks_hashed"], dtype=float)

    if len(t) != len(b):
        raise ValueError(f"{mode}: time_s και blocks_hashed πρέπει να έχουν ίδιο μήκος.")

    # Αν δεν έχεις βάλει δεδομένα ακόμα, το προσπερνάμε
    if len(t) == 0:
        continue

    # cumulative hashed blocks
    cum_b = np.cumsum(b)

    # Γραμμή cumulative
    plt.step(t, cum_b, where="post", label=mode, color=d["color"], linewidth=2)

    # Detection: κατακόρυφη διακεκομμένη + marker στο y που δίνεις
    dt = d["detection_time_s"]
    db = d["detection_blocks"]

    if dt is not None:
        plt.axvline(dt, color=d["color"], linestyle="--", linewidth=1.8)

        if db is not None:
            plt.scatter([dt], [db], color=d["color"], s=60, zorder=5)

            # μικρή ετικέτα (optional)
            plt.text(dt, db, f"  detect", color=d["color"], va="center")

plt.xlabel("Time (s)")
plt.ylabel("Cumulative blocks hashed")
plt.title("Cumulative Blocks Hashed vs Time (per mode) + Detection Time")
plt.legend()
plt.tight_layout()
plt.show()