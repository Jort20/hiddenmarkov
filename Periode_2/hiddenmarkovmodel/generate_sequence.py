# generate_sequence.py

from hmmmodel import HiddenMarkovModel as HMM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Maak een model met 3 toestanden en 4 emissies (bijvoorbeeld kleuren van knikkers)
model = HMM(n_components=3, n_features=4)

# Stel de begintoestanden, overgangswaarschijnlijkheden en emissiekansen in
model.startprob_ = np.array([0.3, 0.4, 0.3])
model.transmat_ = np.array([[0.5, 0.2, 0.3],
                            [0.3, 0.4, 0.3],
                            [0.2, 0.3, 0.5]])
model.emissionprob_ = np.array([[0.1, 0.4, 0.4, 0.1],  # Emissie kansen voor toestand 0
                                [0.3, 0.3, 0.2, 0.2],  # Emissie kansen voor toestand 1
                                [0.2, 0.3, 0.3, 0.2]])  # Emissie kansen voor toestand 2

# Genereer een sequentie van 1200 waarnemingen en toestanden
emissions, states = model.sample(1200)

# Toon een representatie van het model
print(model)

# Toon de eerste paar toestanden en emissies
print("States:", states[:30])
print("Emissions:", emissions[:30])

# Schat overgangswaarschijnlijkheden
transition_counts = np.zeros((model.n_components, model.n_components))
for t in range(1, len(states)):
    transition_counts[states[t-1], states[t]] += 1

estimated_transmat = transition_counts / transition_counts.sum(axis=1, keepdims=True)

# Schat emissiekansen
emission_counts = np.zeros((model.n_components, model.n_features))
for t in range(len(emissions)):
    emission_counts[states[t], emissions[t]] += 1

estimated_emissionprob = emission_counts / emission_counts.sum(axis=1, keepdims=True)

# Print de werkelijke en geschatte matrices
print("Werkelijke overgangsmatrix (transmat):")
print(model.transmat_)
print("Geschatte overgangsmatrix:")
print(estimated_transmat)

print("Werkelijke emissiematrix (emissionprob):")
print(model.emissionprob_)
print("Geschatte emissiematrix:")
print(estimated_emissionprob)

# Plot histogrammen voor toestanden en emissies
plt.figure(figsize=(12, 6))

# Plot voor toestanden
plt.subplot(1, 2, 1)
plt.hist(states, bins=np.arange(model.n_components + 1) - 0.5, edgecolor='black', rwidth=0.8)
plt.title("Histogram van toestanden")
plt.xlabel("Toestand")
plt.ylabel("Aantal voorkomens")

# Plot voor emissies
plt.subplot(1, 2, 2)
plt.hist(emissions, bins=np.arange(model.n_features + 1) - 0.5, edgecolor='black', rwidth=0.8)
plt.title("Histogram van emissies")
plt.xlabel("Emissie")
plt.ylabel("Aantal voorkomens")

plt.tight_layout()
plt.show()

# Heatmaps van werkelijke en geschatte matrices
plt.figure(figsize=(12, 8))

# Werkelijke transmat heatmap
plt.subplot(2, 2, 1)
sns.heatmap(model.transmat_, annot=True, cmap="Blues", cbar=True, fmt=".2f")
plt.title("Werkelijke overgangsmatrix (transmat)")
plt.xlabel("Volgende toestand")
plt.ylabel("Huidige toestand")

# Geschatte transmat heatmap
plt.subplot(2, 2, 2)
sns.heatmap(estimated_transmat, annot=True, cmap="Greens", cbar=True, fmt=".2f")
plt.title("Geschatte overgangsmatrix")
plt.xlabel("Volgende toestand")
plt.ylabel("Huidige toestand")

# Werkelijke emissieprob heatmap
plt.subplot(2, 2, 3)
sns.heatmap(model.emissionprob_, annot=True, cmap="Blues", cbar=True, fmt=".2f")
plt.title("Werkelijke emissiematrix (emissionprob)")
plt.xlabel("Emissie")
plt.ylabel("Toestand")

# Geschatte emissieprob heatmap
plt.subplot(2, 2, 4)
sns.heatmap(estimated_emissionprob, annot=True, cmap="Greens", cbar=True, fmt=".2f")
plt.title("Geschatte emissiematrix")
plt.xlabel("Emissie")
plt.ylabel("Toestand")

plt.tight_layout()
plt.show()
