from hmmmodel import HiddenMarkovModel as HMM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from math import exp, log
from hmmlearn.hmm import CategoricalHMM

# ------------------------------
# Part 1: Model Setup and Sampling
# ------------------------------

# Maak een model met 3 toestanden en 4 emissies (bijvoorbeeld kleuren van knikkers)
model = HMM(n_components=3, n_features=4)

# Stel de begintoestanden, overgangswaarschijnlijkheden en emissiekansen in
startprob = np.array([0.3, 0.4, 0.3])
transmat = np.array([[0.5, 0.2, 0.3],
                     [0.3, 0.4, 0.3],
                     [0.2, 0.3, 0.5]])
emissionprob = np.array([[0.1, 0.4, 0.4, 0.1],  # Emissie kansen voor toestand 0
                         [0.3, 0.3, 0.2, 0.2],  # Emissie kansen voor toestand 1
                         [0.2, 0.3, 0.3, 0.2]])  # Emissie kansen voor toestand 2

model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob

# Genereer een sequentie van 1200 waarnemingen en toestanden
emissions, states = model.sample(1200)

# Toon een representatie van het model
print(model)

# Toon de eerste paar toestanden en emissies
print("States:", states[:30])
print("Emissions:", emissions[:30])

# ------------------------------
# Part 2: Estimating Transition and Emission Probabilities
# ------------------------------

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

# ------------------------------
# Part 3: Plotting Histograms and Heatmaps
# ------------------------------

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

# ------------------------------
# Part 4: Log Probability Calculations
# ------------------------------

# data
X = [1, 2, 2, 3, 3]  # Waarnemingen (0: blauw, 1: geel, 2: groen, 3: rood)
state_sequence = [1, 2, 0, 0, 2]  # Toestanden: tafels 2, 3, 1, 1, 3

# Bereken log-waarschijnlijkheid voor een specifieke toestandsreeks
ln_prob_specific = model.score(X, state_sequence)
print(f"Eigen module (log(p) specifieke reeks): {ln_prob_specific:.3f}")

# Bereken de totale log-waarschijnlijkheid door kansen te sommeren over alle toestandsreeksen
prob_sum = 0.0
for all_states in product(range(3), repeat=len(X)):
    prob_sum += exp(model.score(X, all_states))

ln_prob_total = log(prob_sum)
print(f"Som over ALLE toestandsreeksen (ln(p)): {ln_prob_total:.3f}")

# Vergelijk met hmmlearn
hmm_model = CategoricalHMM(n_components=3, random_state=42)
hmm_model.startprob_ = startprob
hmm_model.transmat_ = transmat
hmm_model.emissionprob_ = emissionprob

hmm_ln_prob = hmm_model.score(np.array(X).reshape(-1, 1))
print(f"hmmlearn score (ln(p)): {hmm_ln_prob:.3f}")

# Resultatenverificatie
assert np.isclose(ln_prob_total, hmm_ln_prob, atol=1e-3), "Scores komen niet overeen!"
print("Resultaten komen overeen met hmmlearn!")
