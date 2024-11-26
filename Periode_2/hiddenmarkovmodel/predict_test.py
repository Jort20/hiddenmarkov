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
real_states = [1, 2, 0, 2, 1]  # Ware toestanden

# Voorspelling met eigen model
predicted_states = model.predict(X)
print("============================= EIGEN MODULE =============================")
print(f"Emissies         : {X}")
print(f"Real states      : {real_states}")
print(f"Predicted states : {predicted_states.tolist()}")

# Bereken percentage correct
accuracy = np.mean(np.array(real_states) == np.array(predicted_states)) * 100
print(f"De overeenkomst tussen ware en voorspelde toestanden is {accuracy:.1f} %.")

# Vergelijk log-waarschijnlijkheden
ln_prob_real = model.score(X, real_states)
ln_prob_predicted = model.score(X, predicted_states)
print(f"Log-waarschijnlijkheid voor real states:      ln(p) = {ln_prob_real:.3f}")
print(f"Log-waarschijnlijkheid voor predicted states: ln(p) = {ln_prob_predicted:.3f}")
if ln_prob_predicted > ln_prob_real:
    print("De voorspelde toestanden hebben een hogere waarschijnlijkheid dan de ware toestanden!")

# Vergelijk met hmmlearn
from hmmlearn.hmm import CategoricalHMM
hmm_model = CategoricalHMM(n_components=3, random_state=42)
hmm_model.startprob_ = startprob
hmm_model.transmat_ = transmat
hmm_model.emissionprob_ = emissionprob

hmm_predicted_states = hmm_model.predict(np.array(X).reshape(-1, 1))
print("=========================== HMMLEARN MODULE ============================")
print(f"Emissies         : {X}")
print(f"Real states      : {real_states}")
print(f"Predicted states : {hmm_predicted_states.tolist()}")
print("De voorspelling komt overeen!" if np.array_equal(predicted_states, hmm_predicted_states) else "De voorspelling komt NIET overeen!")
