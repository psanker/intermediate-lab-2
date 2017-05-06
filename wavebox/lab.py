# -*- coding: utf-8 -*-

#############################################################
# 1. Imports
#############################################################

import numpy as np
import matplotlib.pyplot as plt

import decimal as dec
import math
import os
import shutil
import sys

from astropy import units as u
from astropy import constants as const

from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from numba import jit

from scipy.misc import factorial
from scipy import linalg as lin
from scipy import optimize as opt
from scipy import stats

import sympy as sp

sp.init_printing(use_latex=True, use_unicode=True)

#############################################################
# 2. Constants
#############################################################

PI     = np.pi
TWO_PI = 2*PI

kB   = const.k_B
R    = const.R
NA   = const.N_A
G    = const.G
g    = const.g0
mu    = 1.317e25 * G
R_E  = 6371 * 1000
h    = const.h
hbar = const.hbar
c    = const.c
m_e  = const.m_e
m_n  = const.m_n
m_p  = const.m_p
R_H  = 2.18e-18 * u.J

#############################################################
# 4. Data
#############################################################

data_N = 30
data_m = m_e.value
data_L = 0.0001

#############################################################
# 5. Lab-Specific Functions
#############################################################

@jit
def coeff(N, L):
    ret   = np.empty(N)
    sqrt2 = np.sqrt(2)

    for n in np.arange(1, N + 1):
        if n == 2:
            foo = PI / 2.
        else:
            foo = ((n - 2.)**(-1.)) * np.sin(((n - 2.) * PI) / 2.)

        bar = ((n + 2.)**(-1.)) * np.sin(((n + 2.) * PI) / 2.)

        ret[n - 1] = sqrt2 * (foo - bar)

    return ret

@jit
def energies(t, N, L, m):
    ret = np.empty(N, dtype=complex)
    hb  = hbar.value

    c   = coeff(N, L)
    val = (2. * m)**(-1.) * (hb * PI / L)**(2.)

    for n in np.arange(1, N + 1):
        E = val * (n**2.)

        re = c[n - 1] * np.cos(-1. * (E / hb) * t)
        im = c[n - 1] * np.sin(-1. * (E / hb) * t)
        ret[n - 1] = re + im*1j

    return ret

@jit
def basis(x, N, L):
    ret = np.empty(N)

    for n in np.arange(1, N + 1):
        ret[n - 1] = np.sqrt(1. / L) * np.sin(((PI * n) / (2. * L)) * x)

    return ret

@jit
def build_psi(x, N, L):
    c   = coeff(N, L)
    b   = basis(x, N, L)

    return np.multiply(c, b)

@jit
def psi(x, N, L):
    ret = np.empty(len(x))

    c   = coeff(N, L)

    for i in range(len(x)):
        b      = basis(x[i], N, L)
        ret[i] = np.dot(c, b)

    return ret

@jit
def psi_T(t, x, N, L, m):
    ret = np.empty(len(x), dtype=complex)

    E   = energies(t, N, L, m)

    for i in range(len(x)):
        b      = basis(x[i], N, L)
        psi_x  = np.multiply(c, b)

        ret[i] = np.dot(psi_x, E)

    return ret

@jit
def probability(t, x, n=20, l=5, m=m_e.value):
    wav = psi_T(t, x, n, l, m)

    return np.real(np.multiply(np.conjugate(wav), wav))

def progress_meter(per):
    LEN = 50
    val = per * LEN

    bar = '['

    for i in range(LEN):
        if i <= val:
            bar += '#'
        else:
            bar += '_'

    return (bar + '] %2.0f%%' % (per * 100))

def run_animate():
    # Create cache directory
    directory = 'wavebox'
    cache     = directory + '/dump'

    if not os.path.exists(cache):
        os.makedirs(cache)

    X = np.linspace(0, 2. * data_L, 1000)
    T = np.linspace(0, 2., 200)

    print('Graphs:')

    for i in range(len(T)):
        sys.stdout.write('\r%s' % (progress_meter(float(i) / float(len(T)))))

        plt.figure()
        plt.plot(X, probability(T[i], X, n=data_N, l=data_L, m=data_m))
        plt.savefig('%s/%d.png' % (cache, i))
        plt.close()

    print('\nVideo: ')
    os.system('ffmpeg -framerate 100 -pattern_type glob -i \'%s/*.png\' -c:v libx264 -pix_fmt yuv420p %s/output.mp4' % (cache, directory))

    print('Cleanup:')

    shutil.rmtree(cache)

    print('Done')
