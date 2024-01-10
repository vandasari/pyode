# -*- coding: utf-8 -*-
import numpy as np


def test_func(t, y, p):
    dy = np.zeros((len(y), ))
    dy[0] = p[0] * np.exp(p[1] * t) - p[2] * y[0]
    return dy

def test_params():
    t_range = [0., 2.]
    y_init = [2.]
    params = [4., 0.8, 0.5]
    return t_range, y_init, params


###------------------------------###


def simple_func(t, y, p):
    dy = np.zeros((len(y), ))
    dy[0] = p[0] * np.exp(p[1] * t) - p[2] * y[0]
    return dy

def simple_params():
    t_range = [0., 2.]
    y_init = [2.]
    params = [4., 0.8, 0.5]
    return t_range, y_init, params


###------------------------------###


def vdp_func(t, y, p):
    dy = np.zeros((len(y), ))
    dy[0] = y[1]
    dy[1] = p[0]*(1 - y[0]**2) * y[1] - y[0]
    return dy


def vdp_params():
    t_range = [0., 20.]
    y_init = [2., 0.]
    params = [1.]
    return t_range, y_init, params


###------------------------------###


def brusselator_func(t, y, p):
    dy = np.zeros((len(y),))
    dy[0] = p[0] + p[1] * y[1] * y[0]**2 - p[2] * y[0]
    dy[1] = p[3] * y[0] - p[4] * y[1] * y[0]**2
    return dy

def brusselator_params():
    t_range = [0, 20]
    y_init = [1.5, 3.]
    params = [1., 1., 4., 3., 1.]
    return t_range, y_init, params


###------------------------------###


def lorenz_func(t, y, p):
    dy = np.zeros((len(y), ))
    dy[0] = p[0] * (y[1] - y[0])
    dy[1] = y[0] * (p[1] - y[2]) - y[1]
    dy[2] = y[0] * y[1] - p[2] * y[2]
    return dy

def lorenz_params():
    t_range = [0, 10]
    y_init = [0.4, -0.7, 21.]
    params = [10., 28., 8./3.]
    return t_range, y_init, params
    

###------------------------------###
# Discontinuous equation
# Book: Solving Ordinary Differential Equations I - Nonstiff Problems (1993)
# Pages: 198, Eqs. (6.28) & (6.27)
def coulomb_func(t, y, p):
    dy = np.zeros((len(y),))
    if y[1] > 0:
        a = 4.
    else:
        a = -4.
        
    dy[0] = p[0] * y[1]
    dy[1] = p[1] * y[1] - p[2] * y[0] + p[3] * np.cos(np.pi * t) - a
    return dy

def coulomb_params():
    t_range = [0, 10.]
    y_init = [3., 4.]
    params = [1., -0.2, 1., 2.]
    return t_range, y_init, params


###------------------------------###


def oscillator_func(t, y, p):
    dy = np.zeros((len(y), ))
    dy[0] = ((p[0] + p[1] * y[0]**2) / (1 + y[0]**2 + p[3]*y[1])) - y[0]
    dy[1] = p[5] * (p[2] * y[0] + p[4] - y[1])
    return dy

def oscillator_params():
    t_range = [1, 100]
    y_init = [1, 1]
    params = [1, 5, 4, 1, 0, 0.1]
    return t_range, y_init, params


###------------------------------###

