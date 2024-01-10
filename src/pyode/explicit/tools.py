import numpy as np
from interpolation import SecondOrder


class Interpolate:
    def __init__(self, t, y, yhat, tend):
        self.t = t
        self.y = y
        self.yhat = yhat
        self.tend = tend

    def calculate(self):
        xx = np.array([self.t[-3], self.t[-2], self.t[-1]])
        yy = np.array([self.y[-3, :], self.y[-2, :], self.y[-1, :]])
        yyhat = np.array([self.yhat[-3, :], self.yhat[-2, :], self.yhat[-1, :]])
        n = self.y.shape[1]
        xp = np.array([self.tend])

        res = np.zeros((n,))
        reshat = np.zeros((n,))

        for i in range(n):
            q = SecondOrder(xx, yy[:, i], xp)
            qhat = SecondOrder(xx, yyhat[:, i], xp)
            res[i] = q.quadratic()
            reshat[i] = qhat.quadratic()

        return xp, res, reshat


class Norm:
    def __init__(self, arr, nord):
        self.arr = arr
        self.nord = nord
