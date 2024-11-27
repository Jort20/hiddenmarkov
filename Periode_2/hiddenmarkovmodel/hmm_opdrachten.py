from hmmmodel import HiddenMarkovModel as HMM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from math import exp, log
from hmmlearn.hmm import CategoricalHMM

# ------------------------------
# Modelparameters instellen
# ------------------------------

startprob = np.array([0.3, 0.4, 0.3])
transmat = np.array([[0.5, 0.2, 0.3],
                     [0.3, 0.4, 0.3],
                     [0.2, 0.3, 0.5]])
emissionprob = np.array([[0.1, 0.4, 0.4, 0.1],
                         [0.3, 0.3, 0.2, 0.2],
                         [0.2, 0.3, 0.3, 0.2]])


model = HMM(n_components=3, n_features=4)
model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob

# ------------------------------
# Deel 1: Sample en model inspectie
# ------------------------------


emissions, states = model.sample(1200)


print(model)


print("States:", states[:30])
print("Emissions:", emissions[:30])

# ------------------------------
# Deel 2: Schat overgangs- en emissieprobabiliteiten
# ------------------------------


transition_counts = np.zeros((model.n_components, model.n_components))
for t in range(1, len(states)):
    transition_counts[states[t-1], states[t]] += 1

estimated_transmat = transition_counts / transition_counts.sum(axis=1, keepdims=True)


emission_counts = np.zeros((model.n_components, model.n_features))
for t in range(len(emissions)):
    emission_counts[states[t], emissions[t]] += 1

estimated_emissionprob = emission_counts / emission_counts.sum(axis=1, keepdims=True)


print("Werkelijke overgangsmatrix (transmat):")
print(model.transmat_)
print("Geschatte overgangsmatrix:")
print(estimated_transmat)

print("Werkelijke emissiematrix (emissionprob):")
print(model.emissionprob_)
print("Geschatte emissiematrix:")
print(estimated_emissionprob)

# ------------------------------
# Deel 3: Visualisaties van histogrammen en heatmaps
# ------------------------------


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.hist(states, bins=np.arange(model.n_components + 1) - 0.5, edgecolor='black', rwidth=0.8)
plt.title("Histogram van toestanden")
plt.xlabel("Toestand")
plt.ylabel("Aantal voorkomens")


plt.subplot(1, 2, 2)
plt.hist(emissions, bins=np.arange(model.n_features + 1) - 0.5, edgecolor='black', rwidth=0.8)
plt.title("Histogram van emissies")
plt.xlabel("Emissie")
plt.ylabel("Aantal voorkomens")

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))


plt.subplot(2, 2, 1)
sns.heatmap(model.transmat_, annot=True, cmap="Blues", cbar=True, fmt=".2f")
plt.title("Werkelijke overgangsmatrix (transmat)")
plt.xlabel("Volgende toestand")
plt.ylabel("Huidige toestand")


plt.subplot(2, 2, 2)
sns.heatmap(estimated_transmat, annot=True, cmap="Greens", cbar=True, fmt=".2f")
plt.title("Geschatte overgangsmatrix")
plt.xlabel("Volgende toestand")
plt.ylabel("Huidige toestand")


plt.subplot(2, 2, 3)
sns.heatmap(model.emissionprob_, annot=True, cmap="Blues", cbar=True, fmt=".2f")
plt.title("Werkelijke emissiematrix (emissionprob)")
plt.xlabel("Emissie")
plt.ylabel("Toestand")


plt.subplot(2, 2, 4)
sns.heatmap(estimated_emissionprob, annot=True, cmap="Greens", cbar=True, fmt=".2f")
plt.title("Geschatte emissiematrix")
plt.xlabel("Emissie")
plt.ylabel("Toestand")

plt.tight_layout()
plt.show()

# ------------------------------
# Deel 4: Log waarschijnlijkheden
# ------------------------------
print("                                                 ")

X = [1, 2, 2, 3, 3]  
state_sequence = [1, 2, 0, 0, 2]  


ln_prob_specific = model.score(X, state_sequence)
print(f"Eigen module (log(p) specifieke reeks): {ln_prob_specific:.3f}")


prob_sum = 0.0
for all_states in product(range(3), repeat=len(X)):
    prob_sum += exp(model.score(X, all_states))

ln_prob_total = log(prob_sum)
print(f"Som over ALLE toestandsreeksen (ln(p)): {ln_prob_total:.3f}")


hmm_model = CategoricalHMM(n_components=3, random_state=42)
hmm_model.startprob_ = startprob
hmm_model.transmat_ = transmat
hmm_model.emissionprob_ = emissionprob

hmm_ln_prob = hmm_model.score(np.array(X).reshape(-1, 1))
print(f"hmmlearn score (ln(p)): {hmm_ln_prob:.3f}")


assert np.isclose(ln_prob_total, hmm_ln_prob, atol=1e-3), "Scores komen niet overeen!"
print("Resultaten komen overeen met hmmlearn!")

# ------------------------------
# Deel 5: Voorspellingen en Vergelijkingen
# ------------------------------


X = [1, 2, 0, 3, 2]  
real_states = [1, 2, 0, 2, 1]  


predicted_states = model.predict(X)
print("                                                 ")
print("============================= EIGEN MODULE =============================")
print(f"Emissies         : {X}")
print(f"Real states      : {real_states}")
print(f"Predicted states : {predicted_states.tolist()}")


accuracy = np.mean(np.array(real_states) == np.array(predicted_states)) * 100
print(f"De overeenkomst tussen ware en voorspelde toestanden is {accuracy:.1f} %.")


ln_prob_real = model.score(X, real_states)
ln_prob_predicted = model.score(X, predicted_states)
print(f"Log-waarschijnlijkheid voor real states:      ln(p) = {ln_prob_real:.3f}")
print(f"Log-waarschijnlijkheid voor predicted states: ln(p) = {ln_prob_predicted:.3f}")
if ln_prob_predicted > ln_prob_real:
    print("De voorspelde toestanden hebben een hogere waarschijnlijkheid dan de ware toestanden!")


hmm_predicted_states = hmm_model.predict(np.array(X).reshape(-1, 1))
print("=========================== HMMLEARN MODULE ============================")
print(f"Emissies         : {X}")
print(f"Real states      : {real_states}")
print(f"Predicted states : {hmm_predicted_states.tolist()}")
print("De voorspelling komt overeen!" if np.array_equal(predicted_states, hmm_predicted_states) else "De voorspelling komt NIET overeen!")

# ------------------------------
# Deel 6: Log-waarschijnlijkheden via forward-algoritme
# ------------------------------


log_prob_forward = model.score(X)
print("                                                 ")
print("============================= EIGEN MODULE =============================")
print(f"Emissies : {X}")
print(f"Log-waarschijnlijkheid via forward-algoritme: ln(p) = {log_prob_forward:.3f}")


hmm_log_prob = hmm_model.score(np.array(X).reshape(-1, 1))
print("=============================== HMMLEARN ===============================")
print(f"Emissies : {X}")
print(f"Log-waarschijnlijkheid: ln(p) = {hmm_log_prob:.3f}")


print("De voorspelling komt overeen met de eigen module!" if np.isclose(log_prob_forward, hmm_log_prob, atol=1e-3) else "De voorspelling komt NIET overeen!")
