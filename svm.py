import numpy as np
import qpsolvers

class C_SVM:
    def __init__(self, C):
        self.C = C

    def fit(self, K, y):
        y = 2 * y - 1 # convert {0,1} labels to {-1,1}
        n = len(y)
        q = - y.astype(float)
        G = np.zeros((2 * n, n))
        G[:n, :] = - np.diag(y)
        G[n:, :] = np.diag(y)
        h = np.hstack([np.zeros(n), np.full(n, 2000 * self.C / n)]) # not standard, started with classical SVM...
        
        self.alpha = qpsolvers.solve_qp(K, q, G, h, solver='cvxopt')

        return self

    def predict(self, K):
        return np.where(K @ self.alpha > 0, 1, 0)
    
    def score(self, K, y):
        return np.mean(self.predict(K) == y)