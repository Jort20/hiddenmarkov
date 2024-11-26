from hmmmodel import HiddenMarkovModel as HMM
import numpy as np

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
state_sequence = [1, 2, 0, 2, 1]  # Ware toestanden

# Berekeningen met eigen model
log_prob_with_states = model.score(X, state_sequence)
log_prob_forward = model.score(X)

print("============================= EIGEN MODULE =============================")
print(f"Emissies : {X}")
print(f"Log-waarschijnlijkheid met toestanden:      ln(p) = {log_prob_with_states:.3f}")
print(f"Log-waarschijnlijkheid via forward-algoritme: ln(p) = {log_prob_forward:.3f}")

# Vergelijk met hmmlearn
from hmmlearn.hmm import CategoricalHMM
hmm_model = CategoricalHMM(n_components=3, random_state=42)
hmm_model.startprob_ = startprob
hmm_model.transmat_ = transmat
hmm_model.emissionprob_ = emissionprob

hmm_log_prob = hmm_model.score(np.array(X).reshape(-1, 1))
print("=============================== HMMLEARN ===============================")
print(f"Emissies : {X}")
print(f"Log-waarschijnlijkheid: ln(p) = {hmm_log_prob:.3f}")

# Vergelijk de resultaten
print("De voorspelling komt overeen met de eigen module!" if np.isclose(log_prob_forward, hmm_log_prob, atol=1e-3) else "De voorspelling komt NIET overeen!")
