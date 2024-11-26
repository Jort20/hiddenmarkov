# hmmmodel.py

import numpy as np
from math import log

class HiddenMarkovModel:
    def __init__(self, n_components, n_features):
        """
        Initialiseer het Hidden Markov Model met het aantal toestanden (n_components) en het aantal mogelijke emissies (n_features).
        """
        self.n_components = n_components  # aantal toestanden
        self.n_features = n_features  # aantal mogelijke emissies
        
        # Initialiseren van de matrices met waarden
        self.startprob_ = np.ones(n_components) / n_components  # uniforme start verdeling
        self.transmat_ = np.ones((n_components, n_components)) / n_components  # uniforme overgangsmatrix
        self.emissionprob_ = np.ones((n_components, n_features)) / n_features  # uniforme emissiekansen
        
    def sample(self, n_samples):
        """
        Genereer een sequentie van waarnemingen en toestanden op basis van het HMM.
        
        Parameters:
            n_samples (int): Aantal waarnemingen en toestanden die gegenereerd moeten worden.
        
        Returns:
            emissions (array): Array van waarnemingen.
            states (array): Array van toestanden.
        """
        # Begin met een willekeurige starttoestand
        states = np.zeros(n_samples, dtype=int)
        emissions = np.zeros(n_samples, dtype=int)
        
        # Kies een starttoestand op basis van de startproporties
        states[0] = np.random.choice(self.n_components, p=self.startprob_)
        emissions[0] = np.random.choice(self.n_features, p=self.emissionprob_[states[0], :])
        
        # Genereer de volgende toestanden en waarnemingen
        for t in range(1, n_samples):
            states[t] = np.random.choice(self.n_components, p=self.transmat_[states[t-1], :])
            emissions[t] = np.random.choice(self.n_features, p=self.emissionprob_[states[t], :])
        
        return emissions, states

    def __str__(self):
        """
        Geeft een leesbare string weer van het model.
        """
        return f"HiddenMarkovModel(n_components={self.n_components}, n_features={self.n_features})"

    def __repr__(self):
        """
        Geeft een meer gedetailleerde representatie van het object weer, inclusief de matrices.
        """
        return (f"HiddenMarkovModel(n_components={self.n_components}, n_features={self.n_features}, "
                f"startprob_={self.startprob_}, transmat_={self.transmat_}, "
                f"emissionprob_={self.emissionprob_})")
    
    def score(self, X, state_sequence=None):
        """
        Bereken de log-waarschijnlijkheid van een reeks emissies.
        
        Als een reeks toestanden wordt meegegeven, berekent de methode de log-
        waarschijnlijkheid van de emissies en toestanden samen.
        
        Zonder toestanden gebruikt de methode het forward-algoritme om de log-
        waarschijnlijkheid van de emissies op zich te berekenen.
        
        Parameters:
        - X: iterable van emissies
        - state_sequence: iterable van toestanden (optioneel)
        
        Returns:
        - log_prob: de log-waarschijnlijkheid
        """
        if state_sequence is not None:
            # Bereken log-waarschijnlijkheid met toestanden (deel II)
            log_prob = np.log(self.startprob_[state_sequence[0]])
            log_prob += np.log(self.emissionprob_[state_sequence[0], X[0]])
            for t in range(1, len(X)):
                log_prob += np.log(self.transmat_[state_sequence[t - 1], state_sequence[t]])
                log_prob += np.log(self.emissionprob_[state_sequence[t], X[t]])
            return log_prob

        else:
            # Forward-algoritme
            n_states = self.startprob_.shape[0]
            n_emissions = len(X)

            # Initialisatie (tijdstip t=0)
            forward = np.zeros((n_emissions, n_states))
            forward[0, :] = np.log(self.startprob_) + np.log(self.emissionprob_[:, X[0]])

            # Recursieve stappen
            for t in range(1, n_emissions):
                for s in range(n_states):
                    forward[t, s] = np.logaddexp.reduce(
                        forward[t - 1, :] + np.log(self.transmat_[:, s])
                    ) + np.log(self.emissionprob_[s, X[t]])

            # Terminatie: Som over eindtoestanden
            log_prob = np.logaddexp.reduce(forward[-1, :])
            return log_prob
    
    def predict(self, X):
        """
        Voorspel de meest waarschijnlijke reeks toestanden gegeven de emissies X
        met behulp van het Viterbi-algoritme.

        Parameters:
        - X: iterable van emissies

        Returns:
        - Iterable van toestanden (meest waarschijnlijke toestandsreeks)
        """
        n_states = self.startprob_.shape[0]
        n_emissions = len(X)

        # Initialisatie
        viterbi = np.zeros((n_states, n_emissions))  # Opslag voor scores
        backpointer = np.zeros((n_states, n_emissions), dtype=int)  # Opslag voor backtracking

        # Eerste stap: Initialisatie met begintoestandverdeling
        for s in range(n_states):
            viterbi[s, 0] = np.log(self.startprob_[s]) + np.log(self.emissionprob_[s, X[0]])
            backpointer[s, 0] = 0

        # Recursieve stappen: Overgangen door het model
        for t in range(1, n_emissions):
            for s in range(n_states):
                transition_probs = viterbi[:, t - 1] + np.log(self.transmat_[:, s])
                viterbi[s, t] = np.max(transition_probs) + np.log(self.emissionprob_[s, X[t]])
                backpointer[s, t] = np.argmax(transition_probs)

        # Backtracking: Vind de meest waarschijnlijke toestanden
        states = np.zeros(n_emissions, dtype=int)
        states[-1] = np.argmax(viterbi[:, -1])  # Start bij de laatste stap
        for t in range(n_emissions - 2, -1, -1):
            states[t] = backpointer[states[t + 1], t + 1]

        return states