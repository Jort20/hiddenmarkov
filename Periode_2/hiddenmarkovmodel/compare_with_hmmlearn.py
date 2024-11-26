# compare_with_hmmlearn.py

from hmmlearn.hmm import CategoricalHMM as HMM
import numpy as np

# Maak een HMM van hmmlearn
model_hmmlearn = HMM(n_components=3, n_features=4)

# Stel de parameters in voor hmmlearn
model_hmmlearn.startprob_ = np.array([0.3, 0.4, 0.3])
model_hmmlearn.transmat_ = np.array([[0.5, 0.2, 0.3],
                                     [0.3, 0.4, 0.3],
                                     [0.2, 0.3, 0.5]])
model_hmmlearn.emissionprob_ = np.array([[0.1, 0.4, 0.4, 0.1], 
                                         [0.3, 0.3, 0.2, 0.2], 
                                         [0.2, 0.3, 0.3, 0.2]])

# Genereer een sequentie van 1200 waarnemingen en toestanden met hmmlearn
emissions_hmmlearn, states_hmmlearn = model_hmmlearn.sample(1200)

# Toon de resultaten
print("States hmmlearn:", states_hmmlearn[:30])
print("Emissions hmmlearn:", emissions_hmmlearn[:30])
