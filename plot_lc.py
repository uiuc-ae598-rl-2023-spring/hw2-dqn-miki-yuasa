import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["figure.subplot.bottom"] = 0.15
plt.rcParams["figure.subplot.left"] = 0.15
with open("figures/learning_curve_ablation.pkl", "rb") as f:
    learning_curves = pickle.load(f)[0]

fig, ax = plt.subplots()
labels: list[str] = [
    "with replay and target",
    "no target",
    "no replay",
    "no replay, no target",
]
convolve_value: int = 1000

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
