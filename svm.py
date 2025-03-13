import numpy as np
import qpsolvers

class C_SVM:
    def __init__(self, C):
        self.C = C

    @property
    def _pairwise(self):
        return True

    def fit(self, K, y):
        y = 2*y - 1 # sign
        n = len(y)
        q = -y.astype(float)
        P = K
        G = np.zeros((2 * n, n))
        G[:n, :] = - np.diag(y)
        G[n:, :] = np.diag(y)
        h = np.zeros(2 * n)
        h[n:] = 2000*self.C /n

        alpha = qpsolvers.solve_qp(P, q, G, h, solver='cvxopt')

        self.alpha_ = alpha
        self.fitted_ = True
        self.K_fit_ = K

        return self

    def predict(self, K):
        return (((np.sign(K @ self.alpha_))+ 1) / 2).astype(int)
    
    def score(self, K, y):
        return np.mean(self.predict(K) == y)