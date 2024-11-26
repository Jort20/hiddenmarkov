from hmmmodel import HiddenMarkovModel as HMM
import numpy as np
from itertools import product
from math import exp, log

# Modelparameters
startprob = np.array([0.3, 0.4, 0.3])
transmat = np.array([[0.5, 0.2, 0.3],
                     [0.3, 0.4, 0.3],
                     [0.2, 0.3, 0.5]])
emissionprob = np.array([[0.1, 0.4, 0.4, 0.1],
                         [0.3, 0.3, 0.2, 0.2],
                         [0.2, 0.3, 0.3, 0.2]])

# Maak HMM-model
model = HMM(n_components=3, n_features=4)
model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob

# Testdata
X = [1, 2, 0, 3, 2]  # Waarnemingen
state_sequence = [1, 2, 0, 2, 1]  # Toestanden

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
from hmmlearn.hmm import CategoricalHMM
hmm_model = CategoricalHMM(n_components=3, random_state=42)
hmm_model.startprob_ = startprob
hmm_model.transmat_ = transmat
hmm_model.emissionprob_ = emissionprob

hmm_ln_prob = hmm_model.score(np.array(X).reshape(-1, 1))
print(f"hmmlearn score (ln(p)): {hmm_ln_prob:.3f}")

# Resultatenverificatie
assert np.isclose(ln_prob_total, hmm_ln_prob, atol=1e-3), "Scores komen niet overeen!"
print("Resultaten komen overeen met hmmlearn!")
