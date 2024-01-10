import numpy
import numpy as np

np.seterr(divide="ignore", invalid="ignore")
from explicit.step_size import StepSize
from explicit.initialization import ArrayInitialization
from explicit.estimation import Variables, Approximation
from explicit.tools import Interpolate

###-------------------------###


def norm(x, nord, sc=None):
    if type(nord) == str:
        nord = nord.lower()

    if nord == "inf":
        return max(abs(x))
    elif nord == "-inf":
        return min(abs(x))
    elif nord == 0:
        return sum(x != 0)
    elif nord == 1:
        return sum(abs(x))
    elif nord == 2:
        return (sum(abs(x) ** 2)) ** (1.0 / 2)
    elif nord == "weighted":
        n = len(x)
        return ((1 / n) * sum((x / sc) ** 2)) ** (1 / 2)


def calculateStepSize(rejectStep, hh, hmin, err, p):
    opt = np.sign(1.0 / err) * (np.abs(1.0 / err)) ** (1 / (p + 1))
    if rejectStep == True:
        r = min(0.5, max(0.1, 0.8 * opt))
        hh = max(hmin, hh * r)
    return hh


def RKExplicit(
    func,
    t_range,
    yinit,
    params,
    method="Default",
    abstol=1e-6,
    reltol=1e-3,
    interp="Yes",
):
    method = method.lower()
    interp = interp.lower()

    if reltol == 0.0:
        raise Exception("RelTol cannot be zero")

    atol = abs(abstol)
    rtol = abs(reltol)
    threshold = atol / rtol

    init = ArrayInitialization()

    yinit = init.array_check(yinit)
    t_range = init.array_check(t_range)
    params = init.array_check(params)

    t = t_range[0]
    nord = 2
    n = len(yinit)

    tsol, ysol, yhatsol = init.gen_init_arrays(t, yinit)

    # -- Get Butcher tableau coefficients --#
    vals = Variables(method)
    vals.coefficients()
    b = vals.bt
    bhat = vals.bhat
    p = vals.p

    ya = yinit.copy()
    tspan = t_range[0]

    tdir = np.sign(t_range[-1] - t_range[0])

    # -- Generate initial step size (page 169: Starting Step Size) --#
    ss = StepSize(func, t, yinit, params, method, nord)
    hh, hmax = ss.init_step_v3(t_range, threshold, rtol, p)

    nsteps = 1
    nfailed = 0

    rejectStep = False

    while tspan <= t_range[1]:
        # Step size is bounded by lower (hmin) and upper (hmax)
        hmin = 16 * np.spacing(t)
        hh = min(hmax, max(hmin, hh))
        h = tdir * hh

        noFailed = True  # no failed attempts

        # Loop for moving 1 step forward
        while True:
            yt1 = Approximation(func, t, ya, params, h, method)
            t1, y = yt1.y_approx(b)

            yt2 = Approximation(func, t, ya, params, h, method)
            t2, yhat = yt2.y_approx(bhat)

            # Estimate error
            ydiff = y - yhat
            sc = atol + rtol * max(norm(ya, nord), norm(y, nord))  # Eq. (4.10)
            err = ((1 / n) * (norm(ydiff, nord) / sc) ** 2) ** (
                1 / 2
            )  # norm following Eq (4.11)

            ###--------------------------###

            if err > rtol:
                nfailed += 1

                if hh < hmin:
                    raise ValueError("Integration tolerance not met!")

                if noFailed == True:
                    noFailed = False

                rejectStep = True

                hh = calculateStepSize(rejectStep, hh, hmin, err, p)
                h = tdir * hh
                continue
            else:
                break

            ###--------------------------###

        # If there were no failures in computing a new step
        if noFailed == True and rejectStep == False:
            temp = 1.25 * (err / rtol) ** (1 / (p + 1))
            if temp > 0.2:
                hh = hh / temp
            else:
                hh = 5.0 * hh

        nsteps += 1

        ysol = np.vstack((ysol, y))
        yhatsol = np.vstack((yhatsol, yhat))

        tspan += h
        tsol = np.append(tsol, tspan)

        ya = y
        t = t1
        rejectStep = False

    if interp == "yes":
        intp = Interpolate(tsol, ysol, yhatsol, t_range[1])
        ti, yi, yhi = intp.calculate()
        ysol[-1, :] = yi
        yhatsol[-1, :] = yhi
        tsol[-1] = ti
    elif interp == "no":
        tsol = tsol
        ysol = ysol
        yhatsol = yhatsol

    return (
        tsol,
        ysol,
        yhatsol,
        {
            "total steps": nsteps,
            "failed steps": nfailed,
            "absolute error": atol,
            "relative error": rtol,
        },
    )


###-------------------------###
