import numpy as np

# Overgangswaarschijnlijkheden
cgi_plus = np.array([
    [0.180, 0.274, 0.426, 0.120],  # A -> [A, C, G, T]
    [0.171, 0.368, 0.274, 0.188],  # C -> [A, C, G, T]
    [0.161, 0.339, 0.375, 0.125],  # G -> [A, C, G, T]
    [0.079, 0.355, 0.384, 0.182],  # T -> [A, C, G, T]
])

cgi_minus = np.array([
    [0.300, 0.205, 0.285, 0.210],  # A -> [A, C, G, T]
    [0.322, 0.298, 0.078, 0.302],  # C -> [A, C, G, T]
    [0.248, 0.246, 0.298, 0.208],  # G -> [A, C, G, T]
    [0.177, 0.239, 0.292, 0.292],  # T -> [A, C, G, T]
])

# Mapping nucleotiden naar indices
nucleotiden = ['A', 'C', 'G', 'T']
nuc_to_idx = {nuc: idx for idx, nuc in enumerate(nucleotiden)}

# Log-ruimte overgangswaarschijnlijkheden
log_cgi_plus = np.log(cgi_plus)
log_cgi_minus = np.log(cgi_minus)

# Functie om log-kans te berekenen voor een sequentie
def bereken_log_kans(sequentie, log_transitions):
    log_kans = 0
    for i in range(len(sequentie) - 1):
        huidige = nuc_to_idx[sequentie[i]]
        volgende = nuc_to_idx[sequentie[i + 1]]
        log_kans += log_transitions[huidige, volgende]
    return log_kans

# Functie om een gecombineerd model te maken (8 toestanden)
def genereer_gecombineerd_model():
    gecombineerd_model = np.zeros((8, 8))
    gecombineerd_model[:4, :4] = cgi_plus  # CGI+ naar CGI+
    gecombineerd_model[4:, 4:] = cgi_minus # CGI- naar CGI-
    return gecombineerd_model

# Log-transities voor het gecombineerde model
log_gecombineerd = np.log(genereer_gecombineerd_model())

# Functie voor gecombineerde log-kansen
def bereken_gecombineerde_log_kans(sequentie, log_combined):
    log_kans = 0
    huidige_toestand = nuc_to_idx[sequentie[0]]
    for i in range(len(sequentie) - 1):
        volgende_toestand = nuc_to_idx[sequentie[i + 1]]
        log_kans += log_combined[huidige_toestand, volgende_toestand]
        huidige_toestand = volgende_toestand
    return log_kans

# CGI+ en CGI- gegenereerde sequenties
cgi_plus_seq = "GGACACGCTATGGGAGGGGGGGCCCCAACCCCCGCCGCCCGCTCCTGGACTGCGCCTCCCCCAGGGGCCCGAGAAAAGGGTGCCGGGCGCAGCATGCGCTTGGAAGCCGCGCGGTGTTGATGCCCCCGGACGCCCACAGGGTCGCGACGCGCTTCTACCTCTCCAGACTCTTCTGGACACTCCTGGAAGGGACACCAGGTGTGCCGAGCCTCCCCCTGCGGCCACGTCGCTGGGGCGCCCTGGCCGCGGCAGCCCCCCTGGCAGTGTTCGCCCTAGGAATGTCCCAAGCGTACCTCGGCC"
cgi_minus_seq = "AAGTGGGCCTTCCCTTTGAAAGCTGCGTAGACACCCCATGGATGTAGGAGTTAAAATTCTGCACAACCTAAATCTAATGCGTGGAAGAAGTTCCTGGGATTTTGGGGAAAAGGCCAGGAATGCTATTTTGCACATCTGCCTAGGACCTGGCTTTGAACCAAGGGATAAATTCCTTTGATAGTGAATTAGTGGTAGTCAGAAAGATCAACCAAGCCTCATGCCAGCACATGGGCCCAGCCGTCTGACTTAATCATCTCAATGGAAACAAAGCTCACAGAGAGTGAACCACATGGAGCCTAA"

# Bereken log-kansen van sequenties
cgi_plus_seq_log_cgi_plus = bereken_log_kans(cgi_plus_seq, log_cgi_plus)
cgi_plus_seq_log_cgi_minus = bereken_log_kans(cgi_plus_seq, log_cgi_minus)

cgi_minus_seq_log_cgi_plus = bereken_log_kans(cgi_minus_seq, log_cgi_plus)
cgi_minus_seq_log_cgi_minus = bereken_log_kans(cgi_minus_seq, log_cgi_minus)

print(f"CGI+ seq volgens CGI+ model: ln(p) = {cgi_plus_seq_log_cgi_plus}")
print(f"CGI- seq volgens CGI+ model: ln(p) = {cgi_plus_seq_log_cgi_minus}")
print(f"CGI+ seq volgens CGI- model: ln(p) = {cgi_minus_seq_log_cgi_plus}")
print(f"CGI- seq volgens CGI- model: ln(p) = {cgi_minus_seq_log_cgi_minus}")

# Analyse voor gegeven sequentie
gegeven_sequentie = "TCCCCGCAGGCCATAGCCCGGGACGTCCGACAGCCGGCTGGTGCTGGGGGTAGGCATAATCGCGAGAGCCACCGTCGTCTGTCTGCCTGCTGAGCCTTAG"

log_gegeven_cgi_plus = bereken_log_kans(gegeven_sequentie, log_cgi_plus)
log_gegeven_cgi_minus = bereken_log_kans(gegeven_sequentie, log_cgi_minus)

print(f"Gegeven sequentie volgens CGI+ model: ln(p) = {log_gegeven_cgi_plus}")
print(f"Gegeven sequentie volgens CGI- model: ln(p) = {log_gegeven_cgi_minus}")
