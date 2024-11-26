import numpy as np
from Bio import SeqIO


emissies_cgi = np.array([0.20, 0.30, 0.30, 0.20])  # [A, C, G, T]
emissies_non_cgi = np.array([0.25, 0.25, 0.25, 0.25])  


transities = np.array([
    [0.990, 0.010],  # CGI 
    [0.001, 0.999],  # non-CGI 
])


nucleotiden = ['A', 'C', 'G', 'T']
nuc_to_idx = {nuc: idx for idx, nuc in enumerate(nucleotiden)}


log_transities = np.log(transities)
log_emissies_cgi = np.log(emissies_cgi)
log_emissies_non_cgi = np.log(emissies_non_cgi)


fasta_file = "Casus_HiddenMarkovModel.fasta"
sequence = ""
for record in SeqIO.parse(fasta_file, "fasta"):
    sequence = str(record.seq)


def bereken_log_kans(seq):
    n = len(seq)
    log_alpha = np.zeros((n, 2))  

    
    log_alpha[0, 0] = log_emissies_cgi[nuc_to_idx[seq[0]]]  # CGI
    log_alpha[0, 1] = log_emissies_non_cgi[nuc_to_idx[seq[0]]]  # non-CGI

   
    for i in range(1, n):
        nuc_idx = nuc_to_idx[seq[i]]

        # CGI
        log_alpha[i, 0] = (
            log_emissies_cgi[nuc_idx] +
            max(log_alpha[i-1, 0] + log_transities[0, 0],
                log_alpha[i-1, 1] + log_transities[1, 0])
        )

        # non-CGI
        log_alpha[i, 1] = (
            log_emissies_non_cgi[nuc_idx] +
            max(log_alpha[i-1, 0] + log_transities[0, 1],
                log_alpha[i-1, 1] + log_transities[1, 1])
        )

    return log_alpha


def viterbi(seq):
    n = len(seq)
    log_alpha = bereken_log_kans(seq)
    path = np.zeros(n, dtype=int)  # 0 = CGI, 1 = non-CGI

 
    path[-1] = 0 if log_alpha[-1, 0] > log_alpha[-1, 1] else 1
    for i in range(n - 2, -1, -1):
        if path[i + 1] == 0:  # Als volgende CGI is
            path[i] = 0 if log_alpha[i, 0] + log_transities[0, 0] > log_alpha[i, 1] + log_transities[1, 0] else 1
        else:  # Als volgende non-CGI is
            path[i] = 0 if log_alpha[i, 0] + log_transities[0, 1] > log_alpha[i, 1] + log_transities[1, 1] else 1

    return path


def detecteer_cpg_eilanden(seq, path):
    cgi_positions = []
    start = None
    for i, state in enumerate(path):
        if state == 0 and start is None:
            start = i
        elif state == 1 and start is not None:
            cgi_positions.append((start, i - 1))
            start = None
    if start is not None:  
        cgi_positions.append((start, len(path) - 1))
    return cgi_positions


def gc_gehalte(seq, regions):
    total_gc = 0
    total_len = 0
    for start, end in regions:
        region_seq = seq[start:end + 1]
        gc_count = region_seq.count("G") + region_seq.count("C")
        total_gc += gc_count
        total_len += len(region_seq)
    return total_gc / total_len if total_len > 0 else 0


path = viterbi(sequence)
cgi_regions = detecteer_cpg_eilanden(sequence, path)


cgi_gc = gc_gehalte(sequence, cgi_regions)
non_cgi_regions = [(cgi_regions[i-1][1] + 1, cgi_regions[i][0] - 1) for i in range(1, len(cgi_regions))]
non_cgi_gc = gc_gehalte(sequence, non_cgi_regions)


print(f"Sequentie bevat {len(sequence)} nucleotiden.")
print(f"Gevonden CpG-eilanden:")
for start, end in cgi_regions:
    print(f"  * CpG-eiland van positie {start + 1} tot {end + 1}, lengte {end - start + 1} bp")
print(f"GC-gehalte in alle CpG-eilanden: {cgi_gc:.3f}")
print(f"GC-gehalte in alle non-CGI-regio's: {non_cgi_gc:.3f}")
