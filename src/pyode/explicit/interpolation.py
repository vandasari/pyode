import bisect


class FirstOrder:
    def __init__(self, x, y, xp):
        self.x = x
        self.y = y
        self.xp = xp
        self.n = len(self.x)

    def points(self):
        idx = bisect.bisect_left(self.x, self.xp)
        self.x0, self.x1 = self.x[idx - 1], self.x[idx]
        self.fx0, self.fx1 = self.y[idx - 1], self.y[idx]

    def linear(self):
        return self.fx0 + (self.fx1 - self.fx0) * (self.xp - self.x0) / (
            self.x1 - self.x0
        )

    def lagrange1(self):
        return (self.xp - self.x1) * self.fx0 / (self.x0 - self.x1) + (
            self.xp - self.x0
        ) * self.fx1 / (self.x1 - self.x0)

    def spline1(self):
        m = (self.fx1 - self.fx0) / (self.x1 - self.x0)
        return self.fx0 + m * (self.xp - self.x0)


class SecondOrder:
    def __init__(self, x, y, xp):
        self.x = x
        self.y = y
        self.xp = xp
        self.n = len(self.x)
        self.points()

    def points(self):
        idx = bisect.bisect_left(self.x, self.xp)
        self.x0, self.x1, self.x2 = self.x[idx - 2], self.x[idx - 1], self.x[idx]
        self.fx0, self.fx1, self.fx2 = self.y[idx - 2], self.y[idx - 1], self.y[idx]

    def quadratic(self):
        b0 = self.fx0
        b1 = (self.fx1 - self.fx0) / (self.x1 - self.x0)
        tmp = (self.fx2 - self.fx1) / (self.x2 - self.x1)
        b2 = (tmp - b1) / (self.x2 - self.x0)
        return (
            b0
            + b1 * (self.xp - self.x0)
            + b2 * (self.xp - self.x0) * (self.xp - self.x1)
        )

    def lagrange2(self):
        a0 = (
            (self.xp - self.x1)
            * (self.xp - self.x2)
            / ((self.x0 - self.x1) * (self.x0 - self.x2))
        )
        a1 = (
            (self.xp - self.x0)
            * (self.xp - self.x2)
            / ((self.x1 - self.x0) * (self.x1 - self.x2))
        )
        a2 = (
            (self.xp - self.x0)
            * (self.xp - self.x1)
            / ((self.x2 - self.x0) * (self.x2 - self.x1))
        )
        return a0 * self.fx0 + a1 * self.fx1 + a2 * self.fx2

    def spline2(self):
        pass
