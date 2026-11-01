# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:04:57 2026

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt

def naca0012(chord=100.0, n_points=101):
    # Normalized chord locations
    x = np.linspace(0.0, 1.0, n_points)
    
    # NACA 0012 thickness distribution
    t = 0.12
    yt = 5 * t * ((0.2969*np.sqrt(x)) - (0.1260*x) - (0.3516*(x**2)) + (0.2843*(x**3)) - (0.1015*(x**4)))
    
    # Scale to the requested chord in mm
    x_mm = x * chord
    y_mm = yt * chord
    
    return x_mm, y_mm

x_mm, y_mm = naca0012()

#----------------Plot-----------
plt.figure(figsize=(10,4))
plt.plot(x_mm, y_mm, label='Upper surface')
plt.plot(x_mm, -y_mm, label='Lower surface')
plt.axis('equal')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('NACA 0012 - 100mm chord')
plt.grid(True)
plt.legend()
plt.show()