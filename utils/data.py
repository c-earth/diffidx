import numpy as np

def generate_standard_matrix():
    abc = np.random.random(3)+0.5
    a, b, c = np.sort(abc)
    cos_alpha = (np.random.random(1)[0] - 0.5) * b/c
    cos_beta = (np.random.random(1)[0] - 0.5) * a/c
    cos_gamma = (np.random.random(1)[0] - 0.5) * a/b
    sin_gamma = (1-cos_gamma**2)**0.5
    V = a*b*c*(1-cos_alpha**2-cos_beta**2-cos_gamma**2+2*cos_alpha*cos_beta*cos_gamma)**0.5
    return np.array([[a, 0, 0], [b*cos_gamma, b*sin_gamma, 0], [c*cos_beta, c*(cos_alpha-cos_beta*cos_gamma)/sin_gamma, V/a/b/sin_gamma]])

def generate_perfect_pattern(reciprocal_lattice_matrix, max_q, resolution):
    crosses = np.cross(reciprocal_lattice_matrix[:, [1, 2, 0]], reciprocal_lattice_matrix[:, [2, 0, 1]])
    unit_crosses = crosses / np.linalg.norm(crosses, axis = -1)
    dq = np.abs(np.einsum('ij,ij->i', reciprocal_lattice_matrix, unit_crosses))
    max_hkl = np.floor(max_q/dq)

    hkls = np.stack(np.meshgrid(*[np.arange(-n, n+1) for n in max_hkl])).reshape(3, -1).T
    qs = np.linalg.norm(np.matmul(hkls, reciprocal_lattice_matrix), axis = -1)

    idx_screen = (qs <= max_q)
    qs = qs[idx_screen]
    hkls = hkls[idx_screen]

    idx_sort = np.argsort(qs)
    qs = qs[idx_sort][1:]
    hkls = hkls[idx_sort][1:]

    idx_non_neg = (hkls[:, 0]>0) + (hkls[:, 0]==0) * ((hkls[:, 1]>0) + (hkls[:, 1]==0) * (hkls[:, 2]>0))
    qs = qs[idx_non_neg]
    hkls = hkls[idx_non_neg]


    bin_size = max_q / resolution
    idx_bin = np.floor(qs/bin_size).astype(np.int32)
    pattern = np.zeros(np.max(idx_bin)+1)
    for idx in idx_bin:
        pattern[idx] += 1
    pattern[0] = 0
    return hkls, qs, idx_bin, pattern

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    matrix = generate_standard_matrix()
    hkls, qs, idx_bin, pattern = generate_perfect_pattern(matrix, 2, 1023)
    print(hkls)
    plt.figure()
    plt.plot(pattern)
    plt.savefig('pattern_test.png')