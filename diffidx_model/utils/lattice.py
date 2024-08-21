import numpy as np

class Lattice():
    def __init__(self, a, b, c, alpha, beta, gamma, matrix = None):
        if matrix:
            self.a, self.b, self.c = np.linalg.norm(matrix, axis = 1)
            self.cos_alpha = np.dot(matrix[1], matrix[2])/self.b/self.c
            self.cos_beta = np.dot(matrix[2], matrix[0])/self.c/self.a
            self.cos_gamma = np.dot(matrix[0], matrix[1])/self.a/self.b
        else:
            self.a = a
            self.b = b
            self.c = c
            self.cos_alpha = np.cos(180*alpha/np.pi)
            self.cos_beta = np.cos(180*beta/np.pi)
            self.cos_gamma = np.cos(180*gamma/np.pi)