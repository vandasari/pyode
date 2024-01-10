import numpy as np


class StepSize:
    def __init__(self, f, t0, y0, params, method, nord):
        self.f = f
        self.t0 = t0
        self.y0 = y0
        self.params = params
        self.method = method.lower()
        self.nord = nord
        if (
            self.method == "default"
            or self.method == "rk45"
            or self.method == "rkf45"
            or self.method == "cash-karp"
        ):
            self.p = 4
        elif self.method == "rkv56":
            self.p = 5
        elif self.method == "rkf78" or self.method == "rk78":
            self.p = 7

    def norm(self, x):
        if type(self.nord) == str:
            self.nord = self.nord.lower()
        if self.nord == "inf":
            return max(abs(x))
        elif self.nord == "-inf":
            return min(abs(x))
        elif self.nord == 0:
            return sum(x != 0)
        elif self.nord == 1:
            return sum(abs(x))
        elif self.nord == 2:
            return (sum(abs(x) ** 2)) ** (1.0 / 2)
        else:  # Frobenius norm
            return (sum(abs(x) ** 2)) ** (1.0 / 2)

    # def deriv_0(self):
    #     return self.norm(self.y0)

    # def deriv_1(self):
    #     return self.norm(self.f(self.t0, self.y0, self.params))

    def guess_h0(self, a, b):
        if a < 1e-5 or b < 1e-5:
            return 1e-6
        else:
            return 0.01 * (a / b)

    def explicit_Euler(self, x0):
        return self.y0 + x0 * self.f(self.t0, self.y0, self.params)

    def deriv_2(self, x0, k1):
        num = self.f(self.t0 + x0, k1, self.params) - self.f(
            self.t0, self.y0, self.params
        )
        return self.norm(num) / x0

    def guess_h1(self, u, v, x0):
        if max(u, v) <= 1e-15:
            return max(1e-6, x0 * 1e-3)
        else:
            ph1 = 0.01 / max(u, v)
            return ph1 ** (1 / (self.p + 1))

    def init_step_v1(self):
        # d0 = self.deriv_0()
        # d1 = self.deriv_1()

        d0 = self.norm(self.y0)
        d1 = self.norm(self.f(self.t0, self.y0, self.params))

        h0 = self.guess_h0(d0, d1)

        y1 = self.explicit_Euler(h0)
        d2 = self.deriv_2(h0, y1)
        h1 = self.guess_h1(d1, d2, h0)

        return min(100 * h0, h1)

    # Book: Solving Ordinary Differential Equations I: Nonstiff Problems, pp. 169 (Starting Step Size)
    def init_step_v2(self, abstol, reltol):
        n = len(self.y0)

        # -- Step (a) --#
        sc = abs(abstol) + max(abs(self.y0)) * abs(reltol)
        d0 = ((1 / n) * sum(((self.y0) / sc) ** 2)) ** (1 / 2)
        d1 = ((1 / n) * sum(((self.f(self.t0, self.y0, self.params)) / sc) ** 2)) ** (
            1 / 2
        )

        # -- Step (b): Get a first guess of h --#
        h0 = self.guess_h0(d0, d1)

        # -- Step (c): Perform one explicit Euler step --#
        y1 = self.explicit_Euler(h0)

        # -- Step (d): Estimate the 2nd derivative --#
        yd1 = self.f(self.t0 + h0, y1, self.params)
        yd2 = self.f(self.t0, self.y0, self.params)
        d2 = ((1 / n) * sum(((yd1 - yd2) / sc) ** 2)) ** (1 / 2) / h0

        # -- Step (e): Compute a step size h1 --#
        h1 = self.guess_h1(d1, d2, h0)

        # -- Step (f): Propose a starting step-size --#
        return min(100 * h0, h1)

    # Matlab ODE Suite
    def init_step_v3(self, t_range, threshold, rtol, p):
        dd0 = self.norm(self.y0)  # = d0
        hmax = 1 / 10 * abs(t_range[-1] - t_range[0])
        htspan = abs(t_range[-1] - t_range[0])

        f0 = self.f(self.t0, self.y0, self.params)
        nf0 = self.norm(f0)

        hmin = 16 * np.spacing(self.t0)
        hh = min(hmax, htspan)
        rh = (nf0 / max(dd0, threshold)) / (0.8 * rtol ** (1 / (p + 1)))

        if hh * rh > 1:
            hh = 1 / rh

        hh = max(hh, hmin)
        return hh, hmax
