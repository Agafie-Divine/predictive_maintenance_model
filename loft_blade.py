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

#--------Turn the unit shape into a real blade section
def place (loop, chord, z, twist_deg):
    '''
    Scale to mm, twist about the quarter-chord, set the span position, split into surfaces.
    '''
    X, Y = loop[:, 0] * chord, loop[:, 1] * chord
    if twist_deg:
        a, cx = np.radians(twist_deg), 0.25*chord
        dx = X - cx
        X, Y = cx + dx*np.cos(a) - Y*np.sin(a), dx*np.sin(a) + Y*np.cos(a)
    pts = np.column_stack([X, Y, np.full_like(X, z)])
    ile = int(np.margin(loop[:,0]))
    upper, lower = pts[:ile+1][::1], pts[ile:]
    return pts, upper, lower

def build(spec):
    norm, source = naca4_normalized(spec['code']), f'NACA {spec['code']} 4-digit formula'
    loop, upper, lower = place(norm, spec['chord'], spec['z'], spec['twist'])
    for suffix, arr in (('profile', loop), ('upper', upper), ('lower', lower)):
        np.savetxt(f'{spec['name']}_{suffix}.txt', arr, fmt='%.4f', delimeter='\t')
    d = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    print(f'{spec['name']}: {len(loop)} pts | source: {source} | chord {spec['chord']} mm | z {spec['z']} mm | '
          f'closed: {np.allclose(loop[0], loop[-1])} | zero-length segments: {int((d<1e-6).sum())} | min spacing {d.min():.3f} mm')
    return loop, source

root, root_src = build(ROOT)
tip, tip_src = build(TIP)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(root[:, 0], root[:, 1], label=f'Root: {root_src}, chord {ROOT['chord']:.0f} mm')
ax.plot(tip[:, 0], tip[:, 1], label=f'Tip: {tip_src}, chord {TIP['chord']:.0f} mm')
ax.set_aspect("equal"); ax.grid(True); ax.legend(fontsize=8, loc='upper right')
ax.set_xlabel