import pickle

import matplotlib.pyplot as plt
import numpy as np

with open("figures/learning_curve_ablation.pkl", "rb") as f:
    learning_curves = pickle.load(f)[0]

fig, ax = plt.subplots()
labels: list[str] = [
    "with replay and target",
    "no target",
    "no replay",
    "no replay, no target",
]
convolve_value: int = 200

for learning_curve, label in zip(learning_curves, labels):
    ax.plot(
        np.convolve(
            learning_curve,
            np.ones(convolve_value) / convolve_value,
            mode="valid",
        ),
        label=label,
    )
ax.set_xlabel("Episodes [-]")
ax.set_ylabel("Episodic Reward [-]")
ax.legend()
plt.savefig("figures/learning_curve_ablation.png", dpi=600)
