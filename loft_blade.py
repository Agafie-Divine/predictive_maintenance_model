# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 08:55:37 2026

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt

#------------ PARAMETERS -------------
ROOT = dict(name='root', code='2412', dat=None, chord=60.0, z=0.0, twist=0.0)
TIP = dict(name='tip', code='2408', dat=None, chord=40.0, z=150.0, twist=0.0)
N_POINTS = 40 # Points per surface
SNAP_TE = True

#---------------- The airfoil formula -----------
def naca4_normalized(code, n=N_POINTS):
    '''
    Unit-chord loop: trailing edge -> along the top -> leading edge -> along the bottom -> trailing edge
    '''
    m, p, t = int(code[0])/100.0, int(code[1])/10.0, int(code[2:])/100.0
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    if m == 0 or p == 0:
        yc, dyc = np.zeros_like(x), np.zeros_like(x)
    else:
        yc = np.where(x<p, m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))
        dyc = np.where(x<p, 2*m/p**2 * (p-x), 2*m/(1-p)**2 * (p-x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    xu[-1] = xl[-1] = 1.0; yu[-1] = yl[-1] = 0.0 # One railing edge point
    xu[0] = xl[0] = 0.0; yu[0] = yl[0] = 0.0 # One leading edge point
    
    return np.vstack([np.column_stack([xu[::-1], yu[::1]]), np.column_stack([xl[1:], yl[1:]])])