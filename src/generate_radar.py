import matplotlib.pyplot as plt
import numpy as np

models = ["LogReg", "RandomForest", "XGBoost"]

accuracy = [0.68, 0.81, 0.76]
interpretability = [0.9, 0.5, 0.4]
explainability = [0.85, 0.75, 0.8]

data = [accuracy, interpretability, explainability]
labels = ["Accuracy", "Interpretability", "Explainability"]

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
angles = np.concatenate([angles, [angles[0]]])

fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)

for i, model in enumerate(models):
    values = [data[j][i] for j in range(len(labels))]
    values.append(values[0])
    ax.plot(angles, values, label=model)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.legend()
plt.title("Model Comparison Radar")

plt.savefig("figures/comprehensive_comparison_radar.png", dpi=300)
plt.show()
