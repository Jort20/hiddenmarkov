import numpy as np
import random
from collections import Counter

# Definieer overgangswaarschijnlijkheden
cgi_plus = np.array([
    [0.180, 0.274, 0.426, 0.120],
    [0.170, 0.368, 0.274, 0.188],
    [0.161, 0.339, 0.375, 0.125],
    [0.079, 0.355, 0.384, 0.182]
])

cgi_minus = np.array([
    [0.300, 0.205, 0.285, 0.210],
    [0.322, 0.298, 0.078, 0.302],
    [0.248, 0.246, 0.298, 0.208],
    [0.177, 0.239, 0.292, 0.292]
])

# Mapping nucleotiden
nucleotiden = ['A', 'C', 'G', 'T']

# Functie om een willekeurig nucleotide te selecteren
def kies_start_nucleotide():
    return random.choice(nucleotiden)

# Functie om een sequentie te genereren
def genereer_sequentie(transition_matrix, lengte):
    sequentie = [kies_start_nucleotide()]
    for _ in range(lengte - 1):
        laatste = nucleotiden.index(sequentie[-1])
        volgende = np.random.choice(nucleotiden, p=transition_matrix[laatste])
        sequentie.append(volgende)
    return ''.join(sequentie)

# Functie om kansen op nucleotiden en dinucleotiden te berekenen
def bereken_kansen(sequentie):
    lengte = len(sequentie)
    nucleotide_counts = Counter(sequentie)
    dinucleotide_counts = Counter([sequentie[i:i+2] for i in range(lengte - 1)])
    
    # Kansen op nucleotiden
    nucleotide_probs = {n: nucleotide_counts[n] / lengte for n in nucleotiden}
    
    # Kansen op dinucleotiden
    dinucleotide_probs = {d: dinucleotide_counts[d] / (lengte - 1) for d in dinucleotide_counts}
    
    return nucleotide_probs, dinucleotide_probs

# Functie om ratio's te berekenen
def bereken_ratios(nucleotide_probs, dinucleotide_probs):
    ratios = {}
    for dinucleotide, dinucleotide_prob in dinucleotide_probs.items():
        n1, n2 = dinucleotide
        expected_prob = nucleotide_probs[n1] * nucleotide_probs[n2]
        if expected_prob > 0:
            ratios[dinucleotide] = dinucleotide_prob / expected_prob
        else:
            ratios[dinucleotide] = None
    return ratios

# Genereer sequenties
lengte = 300
cgi_plus_seq = genereer_sequentie(cgi_plus, lengte)
cgi_minus_seq = genereer_sequentie(cgi_minus, lengte)

# Bereken kansen en ratio's
cgi_plus_nucleotide_probs, cgi_plus_dinucleotide_probs = bereken_kansen(cgi_plus_seq)
cgi_plus_ratios = bereken_ratios(cgi_plus_nucleotide_probs, cgi_plus_dinucleotide_probs)

cgi_minus_nucleotide_probs, cgi_minus_dinucleotide_probs = bereken_kansen(cgi_minus_seq)
cgi_minus_ratios = bereken_ratios(cgi_minus_nucleotide_probs, cgi_minus_dinucleotide_probs)

# Resultaten tonen
def print_resultaten(title, sequentie, nucleotide_probs, dinucleotide_probs, ratios):
    print(f"=================================={title}==================================")
    print(f"Sequence:\n{sequentie}\n")
    print(f"Observed nucleotides (ACGT):")
    print([round(nucleotide_probs[n], 3) for n in nucleotiden])
    print("\nObserved dinucleotides:")
    matrix = [[round(dinucleotide_probs.get(n1 + n2, 0), 3) for n2 in nucleotiden] for n1 in nucleotiden]
    for row in matrix:
        print(row)
    print("\nObserved/Expected ratio:")
    ratio_matrix = [[round(ratios.get(n1 + n2, 0), 3) if ratios.get(n1 + n2, 0) is not None else None for n2 in nucleotiden] for n1 in nucleotiden]
    for row in ratio_matrix:
        print(row)
    print("\n")

print_resultaten("CGI+", cgi_plus_seq, cgi_plus_nucleotide_probs, cgi_plus_dinucleotide_probs, cgi_plus_ratios)
print_resultaten("CGI-", cgi_minus_seq, cgi_minus_nucleotide_probs, cgi_minus_dinucleotide_probs, cgi_minus_ratios)
