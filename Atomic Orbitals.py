#!/usr/bin/env python
# coding: utf-8

# example usage
# py h_orbitals_v3.py 2 1 0 1.0 "H 2p_0 "

import sys

if __name__ == "__main__":
    if len(sys.argv) == 4:
        try:
            psicode = sys.argv[1]
            Zeff = float(sys.argv[2])
            fprefix = sys.argv[3]

            print(f"   psicode: {psicode}")
            print(f"      Zeff: {Zeff}")
            print(f"   fprefix: {fprefix}")
            stl_factor = 3.0   # make it X larger for STLs - this seems to work well for Hydrogen
            print(f"stl_factor: {stl_factor}")
            grid_size = 200
            print(f" grid size: {grid_size}")

        except ValueError:
            print("Invalid input. Zeff must be a float.")
            sys.exit(1)
        
    elif len(sys.argv) == 5:
        try:
            psicode = sys.argv[1]
            Zeff = float(sys.argv[2])
            fprefix = sys.argv[3]
            stl_factor = float(sys.argv[4])

            print("   psicode:", psicode)
            print("      Zeff:", Zeff)
            print("   fprefix:", fprefix)
            print("stl_factor:", stl_factor)
            grid_size = 200
            print(f" grid size: {grid_size}")

        except ValueError:
            print("Invalid input. Zeff and stl_factor must be floats.")
            sys.exit(1)
        
    elif len(sys.argv) == 6:
        try:
            psicode = sys.argv[1]
            Zeff = float(sys.argv[2])
            fprefix = sys.argv[3]
            stl_factor = float(sys.argv[4])
            grid_size = int(sys.argv[5])

            print("   psicode:", psicode)
            print("      Zeff:", Zeff)
            print("   fprefix:", fprefix)
            print("stl_factor:", stl_factor)

        except ValueError:
            print("Invalid input. Zeff and stl_factor must be floats. grid_size must be integer.")
            sys.exit(1)
        
    else:
        print("Usage: python h_orbitals_v3.py <psicode> <zeff> <fprefix> [<stl_factor (3)>] [<grid_size> (200)]")
        print("")
        print("Zeff:")
        print("H 1s 1.00")
        print("C 1s 5.67; 2s 3.22; 2p 3.14")
        print("P 1s 14.56; 2s 9.83; 2p 10.96; 3s 5.64; 3p 4.89")
        print("Br 1s 35.00; 2s 34.25; 2p 31.06; 3s 20.22; 3p 19.57; 4s 10.55; 3d 19.56; 4p 9.03")
        sys.exit(1)

import numpy as np
from skimage import measure
import plotly.graph_objects as go
from scipy import integrate
from stl import mesh

import os

# Replace with your desired directory
directory = "C:\\Users\\vince\\OneDrive - Triton College\\Research\\Quantum\\H orbitals\\STLs\\Hydrogen"
# directory = "C:\\Users\\vincenthradil\\OneDrive - Triton College\\Research\\Quantum\\H orbitals\\STLs\\Hydrogen"

# Effective charge
# H 1s 1.00
# C 1s 5.67; 2s 3.22; 2p 3.14
# P 1s 14.56; 2s 9.83; 2p 10.96; 3s 5.64; 3p 4.89
# Br 3d 19.56

# Define constants
a0 = 0.529  # Bohr radius
a0 *= stl_factor

# to get things started
n = 3
l = 1
m = 0

# Calculate parameters
b = Zeff/a0

# grid_size = 200   # trade off resolution for time. Up to 500 works
mesh_threshold_factor = 0.01 # this seems to threshold the density
threshold = 1.0e-9

# In[2]:

print("defining functions ...")
# these are all helper functions
def radial_wavefunction(r, n, l, m):
    if n == 1:
        normalization = 1/np.sqrt(np.pi)
        normtext = "1/np.sqrt(np.pi)"
        exp_part = np.exp(-b*r)
        eptext = "np.exp(-b*r)"
        # l = 0
        bfac = b**(3/2)
        bftext = "b**(3/2)"
        radial_part = 1.0
        rptext = "1.0"
        # m = 0
        norm_adj = 1.0
        natext = "1.0"
    elif n == 2:
        normalization = 1/np.sqrt(32*np.pi)
        normtext = "1/np.sqrt(32*np.pi)"
        exp_part = np.exp(-b*r/2)
        eptext = "np.exp(-b*r/2)"
        if l == 0:
            bfac = b**(3/2)
            bftext = "b**(3/2)"
            radial_part = 2-b*r
            rptext = "2-b*r"
            # m = 0
            norm_adj = 1.0
            natext = "1.0"
        else:
            bfac = b**(5/2)
            bftext = "b**(5/2)"
            radial_part = r
            rptext = "r"
            # m = any
            norm_adj = 1.0
            natext = "1.0"
    elif n == 3:
        normalization = 1/(81*np.sqrt(np.pi))
        normtext = "1/(81*np.sqrt(np.pi))"
        exp_part = np.exp(-b*r/3)
        eptext = "np.exp(-b*r/3)"
        if l == 0:
            bfac = b**(3/2)
            bftext = "b**(3/2)"
            radial_part = 27-18*b*r+2*(b*r)**2
            rptext = "27-18*b*r+2*(b*r)**2"
            # m = 0
            norm_adj = 1/np.sqrt(3)
            natext = "1/np.sqrt(3)"
        elif l == 1:
            bfac = b**(5/2)
            bftext = "b**(5/2)"
            radial_part = (6-b*r)*r
            rptext = "(6-b*r)*r"
            # m = any
            norm_adj = np.sqrt(2)
            natext = "np.sqrt(2)"
        else:
            bfac = b**(7/2)
            bftext = "b**(7/2)"
            radial_part = r**2
            rptext = "r**2"
            if m == -2:
                norm_adj = 1/np.sqrt(2)
                natext = "1/np.sqrt(2)"
            elif m == 2:
                norm_adj = 1/np.sqrt(2)
                natext = "1/np.sqrt(2)"
            elif m == -1:
                norm_adj = np.sqrt(2)
                natext = "np.sqrt(2)"
            elif m == 1:
                norm_adj = np.sqrt(2)
                natext = "np.sqrt(2)"
            else:
                norm_adj = 1/np.sqrt(6)
                natext = "1/np.sqrt(6)"
    elif n == 4:
        normalization = 1/(512*np.sqrt(np.pi))
        normtext = "1/(512*np.sqrt(np.pi))"
        exp_part = np.exp(-b*r/4)
        eptext = "np.exp(-b*r/4)"
        if l == 0:
            bfac = b**(3/2)
            bftext = "b**(3/2)"
            radial_part = 192 - 144*b*r + 24*(b*r)**2 - (b*r)**3
            rptext = "192 - 144*b*r + 24*(b*r)**2 - (b*r)**3"
            # m = 0
            norm_adj = 1.0/3.0
            natext = "1/3"
        elif l == 1:
            bfac = b**(5/2)
            bftext = "b**(5/2)"
            radial_part = (80-20*b*r+(b*r)**2)*r
            rptext = "(80-20*b*r+(b*r)**2)*r"
            # m = any
            norm_adj = np.sqrt(5)
            natext = "np.sqrt(5)"
        elif l == 2:
            bfac = b**(7/2)
            bftext = "b**(7/2)"
            radial_part = (12-b*r)*r**2
            rptext = "(12-b*r)*r**2"
            if m == -2:
                norm_adj = 1/np.sqrt(3)
                natext = "1/np.sqrt(3)"
            elif m == 2:
                norm_adj = 1/np.sqrt(3)
                natext = "1/np.sqrt(3)"
            elif m == -1:
                norm_adj = np.sqrt(12)
                natext = "np.sqrt(12)"
            elif m == 1:
                norm_adj = np.sqrt(12)
                natext = "np.sqrt(12)"
            else:
                norm_adj = 1.0/6.0
                natext = "1/6"
                    
        else:
            bfac = b**(9/2)
            bftext = "b**(7/2)"
            radial_part = r**3
            rptext = "r**3"
            if m == -3:
                norm_adj = 1/(6*np.sqrt(2))
                natext = "1/(6*np.sqrt(2))"
            elif m == 3:
                norm_adj = 1/(6*np.sqrt(2))
                natext = "1/(6*np.sqrt(2))"
            if m == -2:
                norm_adj = np.sqrt(3)/(6*np.sqrt(2))
                natext = "np.sqrt(3)/(6*np.sqrt(2))"
            elif m == 2:
                norm_adj = np.sqrt(3)/(6*np.sqrt(2))
                natext = "np.sqrt(3)/(6*np.sqrt(2))"
            elif m == -1:
                norm_adj = np.sqrt(3)/(6*np.sqrt(10))
                natext = "np.sqrt(3)/(6*np.sqrt(10))"
            elif m == 1:
                norm_adj = np.sqrt(3)/(6*np.sqrt(10))
                natext = "np.sqrt(3)/(6*np.sqrt(10))"
            else:
                norm_adj = 1/(6*np.sqrt(5))
                natext = "1/6"
                    
    else:
        print(f"n is too large: {n}")
        sys.exit()
  
    # print("radial wavefunction")
    # print("      normalization: " + normtext)
    # print("                exp: " + eptext)
    # print("               bfac: " + bftext)
    # print("                 rp: " + rptext)
    # print("           norm_adj: " + natext)

    return normalization * exp_part * bfac * radial_part * norm_adj

# Define angular wavefunction
def angular_wavefunction(theta, phi, l, m):
    if l == 0 :
        angular_part = np.ones_like(theta)
    elif l == 1 :
        if m == -1:
            angular_part = np.sin(theta)*np.cos(phi)
        elif m == 1:
            angular_part = np.sin(theta)*np.sin(phi)
        else: # m == 0 or |m| > 1
            m = 0
            angular_part = np.cos(theta)
    elif l == 2 :
        if m == -2:
            angular_part = ((np.sin(theta))**2)*np.sin(2*phi)
        elif m == 2:
            angular_part = ((np.sin(theta))**2)*np.cos(2*phi)
        elif m == -1:
            angular_part = np.sin(theta)*np.cos(theta)*np.sin(phi)
        elif m == 1:
            angular_part = np.sin(theta)*np.cos(theta)*np.cos(phi)
        else:   # m == 0 or |m| > 2
            angular_part = 3*((np.cos(theta))**2)-1
    elif l == 3:
        if m == -3:
            angular_part = ((np.sin(theta))**3)*np.sin(3*phi)
        elif m == 3:
            angular_part = ((np.sin(theta))**3)*np.cos(3*phi)
        elif m == -2:
            angular_part = ((np.sin(theta))**2)*np.cos(theta)*np.sin(2*phi)
        elif m == 2:
            angular_part = ((np.sin(theta))**2)*np.cos(theta)*np.cos(2*phi)
        elif m == -1:
            angular_part = np.sin(theta)*(5*((np.cos(theta))**2)-1)*np.sin(phi)
        elif m == 1:
            angular_part = np.sin(theta)*(5*((np.cos(theta))**2)-1)*np.cos(phi)
        else:   # m == 0 or |m| > 3
            angular_part = 5*((np.cos(theta))**3)-3*np.cos(theta)
    else: # n > 4
        angular_part = np.ones_like(theta)     
    return angular_part

# Electron density function
def wavefunction(x, y, z, n, l, m):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / (r + 1e-10))  # Avoid division by zero
    phi = np.arctan2(y, x)
    print(f"     in wavefunction getting psi {n} {l} {m}")
    psi = radial_wavefunction(r, n, l, m) * angular_wavefunction(theta, phi, l, m)
    print("     done")
    return psi
    
def getpsi(X, Y, Z, psicode):
    print(f"Getting {psicode}")
    
    match psicode:
# # Wavefunction definitions
        case "psi100":
            # 1s
            psi = wavefunction(X, Y, Z, 1,0,0)

        case "psi200":
            # 2s
            psi = wavefunction(X, Y, Z, 2,0,0)

        case "psi21m1":
            # 2px
            psi = wavefunction(X, Y, Z, 2,1,-1)
        case "psi210":
            # 2pz
            psi = wavefunction(X, Y, Z, 2,1,0)
        case "psi211":
            # 2py
            psi = wavefunction(X, Y, Z, 2,1,1)

        case "psi300":
            # 3s
            psi = wavefunction(X, Y, Z, 3,0,0)

        case "psi31m1":
            # 3px
            psi = wavefunction(X, Y, Z, 3,1,-1)
        case "psi310":
            # 3pz
            psi = wavefunction(X, Y, Z, 3,1,0)
        case "psi311":
            # 3py
            psi = wavefunction(X, Y, Z, 3,1,1)

        case "psi32m2":
            # 3dxy
            psi = wavefunction(X, Y, Z, 3,2,-2)
        case "psi32m1":
            # 3dxz
            psi = wavefunction(X, Y, Z, 3,2,-1)
        case "psi320":
            # 3dz2
            psi = wavefunction(X, Y, Z, 3,2,0)
        case "psi321":
            # 3dyz
            psi = wavefunction(X, Y, Z, 3,2,1)
        case "psi322":
            # 3dx2-y2
            psi = wavefunction(X, Y, Z, 3,2,2)

        case "psi400":
            # 4s
            psi = wavefunction(X, Y, Z, 4, 0 ,0)
            
        case "psi41m1":
            # 4px
            psi = wavefunction(X, Y, Z, 4, 1 ,-1)
        case "psi410":
            # 4pz
            psi = wavefunction(X, Y, Z, 4, 1 ,0)
        case "psi411":
            # 4py
            psi = wavefunction(X, Y, Z, 4, 1 ,1)
            
        case "psi42m2":
            # 4dxy
            psi = wavefunction(X, Y, Z, 4, 2 ,-2)
        case "psi42m1":
            # 4dxz
            psi = wavefunction(X, Y, Z, 4, 2 ,-1)
        case "psi420":
            # 4dz2
            psi = wavefunction(X, Y, Z, 4, 2 ,0)
        case "psi421":
            # 4dyz
            psi = wavefunction(X, Y, Z, 4, 2 ,1)
        case "psi422":
            # 4dx2-y2
            psi = wavefunction(X, Y, Z, 4, 2, 2)

        case "psi43m3":
            # 4fy3x2-y2
            psi = wavefunction(X, Y, Z, 4, 3 ,-3)
        case "psi43m2":
            # 4fxyz
            psi = wavefunction(X, Y, Z, 4, 3 ,-2)
        case "psi43m1":
            # 4fyz2
            psi = wavefunction(X, Y, Z, 4, 3 ,-1)
        case "psi430":
            # 4fz3
            psi = wavefunction(X, Y, Z, 4, 3 ,0)
        case "psi431":
            # 4fxz2
            psi = wavefunction(X, Y, Z, 4, 3 ,1)
        case "psi432":
            # 4fzx2-y2
            psi = wavefunction(X, Y, Z, 4, 3, 2)
        case "psi433":
            # 4fxx2-3y2
            psi = wavefunction(X, Y, Z, 4, 3, 3)

# # Hybrid wavefunctions

# # 2sp
        case "psi2sp_1":
            psi = np.sqrt(1/2) * (getpsi(X, Y, Z, "psi200") + getpsi(X, Y, Z, "psi210"))
        case "psi2sp_2":
            psi = np.sqrt(1/2) * (getpsi(X, Y, Z, "psi200") - getpsi(X, Y, Z, "psi210"))

# # 2sp2
        case "psi2sp2_1":
            psi = np.sqrt(1/3) * (getpsi(X, Y, Z, "psi200") + np.sqrt(2)*getpsi(X, Y, Z, "psi210"))
        case "psi2sp2_2":
            psi = np.sqrt(1/3) * (getpsi(X, Y, Z, "psi200") - np.sqrt(1/2)*getpsi(X, Y, Z, "psi210") + np.sqrt(3/2)*getpsi(X, Y, Z, "psi21m1"))
        case "psi2sp2_3":
            psi = np.sqrt(1/3) * (getpsi(X, Y, Z, "psi200") - np.sqrt(1/2)*getpsi(X, Y, Z, "psi210") - np.sqrt(3/2)*getpsi(X, Y, Z, "psi21m1"))

# # 2sp3
        case "psi2sp3_1":
            psi = (1/2) * (getpsi(X, Y, Z, "psi200") + getpsi(X, Y, Z, "psi21m1") + getpsi(X, Y, Z, "psi211") + getpsi(X, Y, Z, "psi210"))
        case "psi2sp3_2":
            psi = (1/2) * (getpsi(X, Y, Z, "psi200") + getpsi(X, Y, Z, "psi21m1") - getpsi(X, Y, Z, "psi211") - getpsi(X, Y, Z, "psi210"))
        case "psi2sp3_3":
            psi = (1/2) * (getpsi(X, Y, Z, "psi200") - getpsi(X, Y, Z, "psi21m1") + getpsi(X, Y, Z, "psi211") - getpsi(X, Y, Z, "psi210"))
        case "psi2sp3_4":
            psi = (1/2) * (getpsi(X, Y, Z, "psi200") - getpsi(X, Y, Z, "psi21m1") - getpsi(X, Y, Z, "psi211") + getpsi(X, Y, Z, "psi210"))

# # 3sp
        case "psi3sp_1":
            psi = np.sqrt(1/2) * (getpsi(X, Y, Z, "psi300") + getpsi(X, Y, Z, "psi310"))
        case "psi3sp_2":
            psi = np.sqrt(1/2) * (getpsi(X, Y, Z, "psi300") - getpsi(X, Y, Z, "psi310"))

# # 3sp2
        case "psi3sp2_1":
            psi = np.sqrt(1/3) * (getpsi(X, Y, Z, "psi300") + np.sqrt(2)*getpsi(X, Y, Z, "psi310"))
        case "psi3sp2_2":
            psi = np.sqrt(1/3) * (getpsi(X, Y, Z, "psi300") - np.sqrt(1/2)*getpsi(X, Y, Z, "psi310") + np.sqrt(3/2)*getpsi(X, Y, Z, "psi31m1"))
        case "psi3sp2_3":
            psi = np.sqrt(1/3) * (getpsi(X, Y, Z, "psi300") - np.sqrt(1/2)*getpsi(X, Y, Z, "psi310") - np.sqrt(3/2)*getpsi(X, Y, Z, "psi31m1"))

# # 3sp3
        case "psi3sp3_1":
            psi = (1/2) * (getpsi(X, Y, Z, "psi300") + getpsi(X, Y, Z, "psi31m1") + getpsi(X, Y, Z, "psi311") + getpsi(X, Y, Z, "psi310"))
        case "psi3sp3_2":
            psi = (1/2) * (getpsi(X, Y, Z, "psi300") + getpsi(X, Y, Z, "psi31m1") - getpsi(X, Y, Z, "psi311") - getpsi(X, Y, Z, "psi310"))
        case "psi3sp3_3":
            psi = (1/2) * (getpsi(X, Y, Z, "psi300") - getpsi(X, Y, Z, "psi31m1") + getpsi(X, Y, Z, "psi311") - getpsi(X, Y, Z, "psi310"))
        case "psi3sp3_4":
            psi = (1/2) * (getpsi(X, Y, Z, "psi300") - getpsi(X, Y, Z, "psi31m1") - getpsi(X, Y, Z, "psi311") + getpsi(X, Y, Z, "psi310"))

# # 3sp3d
        case "psi3sp3d_1":
            psi = np.sqrt(1/6) * (np.sqrt(2)*getpsi(X, Y, Z, "psi300") + 2*getpsi(X, Y, Z, "psi31m1"))
        case "psi3sp3d_2":
            psi = np.sqrt(1/6) * (np.sqrt(2)*getpsi(X, Y, Z, "psi300") - getpsi(X, Y, Z, "psi31m1") + np.sqrt(3)*getpsi(X, Y, Z, "psi311"))
        case "psi3sp3d_3":
            psi = np.sqrt(1/6) * (np.sqrt(2)*getpsi(X, Y, Z, "psi300") - getpsi(X, Y, Z, "psi31m1") - np.sqrt(3)*getpsi(X, Y, Z, "psi311"))
        case "psi3sp3d_4":
            psi = np.sqrt(1/6) * (np.sqrt(3)*getpsi(X, Y, Z, "psi310") + np.sqrt(3)*getpsi(X, Y, Z, "psi320"))
        case "psi3sp3d_5":
            psi = np.sqrt(1/6) * (np.sqrt(3)*getpsi(X, Y, Z, "psi310") - np.sqrt(3)*getpsi(X, Y, Z, "psi320"))

# # 3sp3d2
        case "psi3sp3d2_1":
            psi = np.sqrt(1/6) * (getpsi(X, Y, Z, "psi300") + np.sqrt(3)*getpsi(X, Y, Z, "psi310") + np.sqrt(2)*getpsi(X, Y, Z, "psi320"))
        case "psi3sp3d2_2":
            psi = np.sqrt(1/6) * (getpsi(X, Y, Z, "psi300") + np.sqrt(3)*getpsi(X, Y, Z, "psi31m1") - np.sqrt(1/2)*getpsi(X, Y, Z, "psi320") + np.sqrt(3/2)*getpsi(X, Y, Z, "psi322"))
        case "psi3sp3d2_3":
            psi = np.sqrt(1/6) * (getpsi(X, Y, Z, "psi300") + np.sqrt(3)*getpsi(X, Y, Z, "psi311") - np.sqrt(1/2)*getpsi(X, Y, Z, "psi320") - np.sqrt(3/2)*getpsi(X, Y, Z, "psi322"))
        case "psi3sp3d2_4":
            psi = np.sqrt(1/6) * (getpsi(X, Y, Z, "psi300") - np.sqrt(3)*getpsi(X, Y, Z, "psi31m1") - np.sqrt(1/2)*getpsi(X, Y, Z, "psi320") + np.sqrt(3/2)*getpsi(X, Y, Z, "psi322"))
        case "psi3sp3d2_5":
            psi = np.sqrt(1/6) * (getpsi(X, Y, Z, "psi300") - np.sqrt(3)*getpsi(X, Y, Z, "psi311") - np.sqrt(1/2)*getpsi(X, Y, Z, "psi320") - np.sqrt(3/2)*getpsi(X, Y, Z, "psi322"))
        case "psi3sp3d2_6":
            psi = np.sqrt(1/6) * (getpsi(X, Y, Z, "psi300") - np.sqrt(3)*getpsi(X, Y, Z, "psi310") + np.sqrt(2)*getpsi(X, Y, Z, "psi320"))
            
        case _:
            print(f"No match for psicode: {psicode}")
            sys.exit()
        
    return psi

def electron_density(psi):
    return psi**2

def generate_mesh(density, threshold):
    verts, faces, _, _ = measure.marching_cubes(density, threshold, spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    return verts, faces

# Create a 3D grid of points

print(f"creating a grid {grid_size} points ...")

# first using standard max_r value
max_r = 2.0 * n**2 * a0  # Maximum radius
x = np.linspace(-max_r, max_r, grid_size)
y = np.linspace(-max_r, max_r, grid_size)
z = np.linspace(-max_r, max_r, grid_size)
X, Y, Z = np.meshgrid(x, y, z)

r = np.sqrt(x**2 + y**2 + z**2)
# rwf = radial_wavefunction(r, n, l, m)
# radial_distribution = r**2 * rwf**2

# integral_radial_distribution = integrate.cumulative_trapezoid(radial_distribution, r, initial=0)

# index = np.argmin(np.abs(integral_radial_distribution + threshold))
# r_value = r[index]

# fig = go.Figure()

# fig.add_trace(go.Scatter(x=r, y=rwf, mode='lines', name='Radial Function'))
# fig.add_trace(go.Scatter(x=r, y=radial_distribution, mode='lines', name='Radial Distribution'))
# fig.add_trace(go.Scatter(x=r, y=integral_radial_distribution, mode='lines', name='Integral Radial Distribution'))
# fig.add_shape(go.layout.Shape(type="line", x0=r_value, y0=0, x1=r_value, y1=1, line=dict(color="Red", width=2)))

# ymin = min(min(rwf),min(radial_distribution),min(integral_radial_distribution))
# ymax = max(max(rwf),max(radial_distribution),max(integral_radial_distribution))

# fig.update_layout(title='Radial Distribution and Integral Radial Distribution',
                  # xaxis_title='r',
                  # yaxis_title='Probability Density',
                  # yaxis_range=[ymin,ymax])

# fig.show()

# Recreate a 3D grid of points using a good rvalue
# print("re-creating a grid ...")
# max_r = 1.5*r_value  # Maximum radius 50% bigger
# print(f"max_r is: {max_r}")

# x = np.linspace(-max_r, max_r, grid_size)
# y = np.linspace(-max_r, max_r, grid_size)
# z = np.linspace(-max_r, max_r, grid_size)
# X, Y, Z = np.meshgrid(x, y, z)

# rwf = radial_wavefunction(r, n, l, m)
# radial_distribution = r**2 * rwf**2

# integral_radial_distribution = integrate.cumulative_trapezoid(radial_distribution, r, initial=0)
# index = np.argmin(np.abs(integral_radial_distribution + threshold))
# r_value = r[index]

# fig = go.Figure()

# fig.add_trace(go.Scatter(x=r, y=rwf, mode='lines', name='Radial Function'))
# fig.add_trace(go.Scatter(x=r, y=radial_distribution, mode='lines', name='Radial Distribution'))
# fig.add_trace(go.Scatter(x=r, y=integral_radial_distribution, mode='lines', name='Integral Radial Distribution'))
# fig.add_shape(go.layout.Shape(type="line", x0=r_value, y0=0, x1=r_value, y1=1, line=dict(color="Red", width=2)))

# ymin = min(min(rwf),min(radial_distribution),min(integral_radial_distribution))
# ymax = max(max(rwf),max(radial_distribution),max(integral_radial_distribution))

# fig.update_layout(title='NEW Radial Distribution and Integral Radial Distribution',
                  # xaxis_title='r',
                  # yaxis_title='Probability Density',
                  # yaxis_range=[ymin,ymax])

# fig.show()

# # Wavefunction definitions
# psi100 = wavefunction(X, Y, Z, 1,0,0)

# psi200 = wavefunction(X, Y, Z, 2,0,0)

# psi21m1 = wavefunction(X, Y, Z, 2,1,-1)
# psi210 = wavefunction(X, Y, Z, 2,1,0)
# psi211 = wavefunction(X, Y, Z, 2,1,1)

# psi300 = wavefunction(X, Y, Z, 3,0,0)

# psi31m1 = wavefunction(X, Y, Z, 3,1,-1)
# psi310 = wavefunction(X, Y, Z, 3,1,0)
# psi311 = wavefunction(X, Y, Z, 3,1,1)

# psi32m2 = wavefunction(X, Y, Z, 3,2,-2)
# psi32m1 = wavefunction(X, Y, Z, 3,2,-1)
# psi320 = wavefunction(X, Y, Z, 3,2,0)
# psi321 = wavefunction(X, Y, Z, 3,2,1)
# psi322 = wavefunction(X, Y, Z, 3,2,2)

# # Hybrid wavefunctions

# # 2sp
# psi2sp_1 = np.sqrt(1/2) * (psi200 + psi210)
# psi2sp_2 = np.sqrt(1/2) * (psi200 - psi210)

# # 2sp2
# psi2sp2_1 = np.sqrt(1/3) * (psi200 + np.sqrt(2)*psi210)
# psi2sp2_2 = np.sqrt(1/3) * (psi200 - np.sqrt(1/2)*psi210 + np.sqrt(3/2)*psi21m1)
# psi2sp2_3 = np.sqrt(1/3) * (psi200 - np.sqrt(1/2)*psi210 - np.sqrt(3/2)*psi21m1)

# # 2sp3
# psi2sp3_1 = (1/2) * (psi200 + psi21m1 + psi211 + psi210)
# psi2sp3_2 = (1/2) * (psi200 + psi21m1 - psi211 - psi210)
# psi2sp3_3 = (1/2) * (psi200 - psi21m1 + psi211 - psi210)
# psi2sp3_4 = (1/2) * (psi200 - psi21m1 - psi211 + psi210)

# # 3sp
# psi3sp_1 = np.sqrt(1/2) * (psi300 + psi310)
# psi3sp_2 = np.sqrt(1/2) * (psi300 - psi310)

# # 3sp2
# psi3sp2_1 = np.sqrt(1/3) * (psi300 + np.sqrt(2)*psi310)
# psi3sp2_2 = np.sqrt(1/3) * (psi300 - np.sqrt(1/2)*psi310 + np.sqrt(3/2)*psi31m1)
# psi3sp2_3 = np.sqrt(1/3) * (psi300 - np.sqrt(1/2)*psi310 - np.sqrt(3/2)*psi31m1)

# # 3sp3
# psi3sp3_1 = (1/2) * (psi300 + psi31m1 + psi311 + psi310)
# psi3sp3_2 = (1/2) * (psi300 + psi31m1 - psi311 - psi310)
# psi3sp3_3 = (1/2) * (psi300 - psi31m1 + psi311 - psi310)
# psi3sp3_4 = (1/2) * (psi300 - psi31m1 - psi311 + psi310)

# # 3sp3d
# psi3sp3d_1 = np.sqrt(1/6) * (np.sqrt(2)*psi300 + 2*psi311)
# psi3sp3d_2 = np.sqrt(1/6) * (np.sqrt(2)*psi300 - psi31m1 + np.sqrt(3)*psi311)
# psi3sp3d_3 = np.sqrt(1/6) * (np.sqrt(2)*psi300 - psi31m1 - np.sqrt(3)*psi311)
# psi3sp3d_4 = np.sqrt(1/6) * (np.sqrt(3)*psi300 + np.sqrt(3)*psi320)
# psi3sp3d_5 = np.sqrt(1/6) * (np.sqrt(3)*psi300 - np.sqrt(3)*psi320)

# # 3sp3d2
# psi3sp3d2_1 = np.sqrt(1/6) * (psi300 + np.sqrt(3)*psi310 + np.sqrt(2)*psi320)
# psi3sp3d2_2 = np.sqrt(1/6) * (psi300 + np.sqrt(3)*psi31m1 - np.sqrt(1/2)*psi320 + np.sqrt(3/2)*psi32m1)
# psi3sp3d2_3 = np.sqrt(1/6) * (psi300 + np.sqrt(3)*psi311 - np.sqrt(1/2)*psi320 - np.sqrt(3/2)*psi32m1)
# psi3sp3d2_4 = np.sqrt(1/6) * (psi300 - np.sqrt(3)*psi31m1 - np.sqrt(1/2)*psi320 + np.sqrt(3/2)*psi32m1)
# psi3sp3d2_5 = np.sqrt(1/6) * (psi300 - np.sqrt(3)*psi311 - np.sqrt(1/2)*psi320 - np.sqrt(3/2)*psi32m1)
# psi3sp3d2_6 = np.sqrt(1/6) * (psi300 - np.sqrt(3)*psi310 + np.sqrt(2)*psi320)

# # psi = one of the above - include all AOs needed for the HOs
# # psi = wavefunction(X, Y, Z, 3,0,0)

print("calculating psi ...")
# R = np.sqrt(X**2 + Y**2 + Z**2)

# print("getting the rwf")
# rwf = radial_wavefunction(R, n, l, m)

# # rwf_copy = np.copy(rwf)
# # indices = np.where(R > r_value)
# # rwf_copy[indices] = 0.0

# # rwf = rwf_copy

# print("making angles")
# theta = np.arccos(Z / (R + 1e-10))  # Avoid division by zero
# phi = np.arctan2(Y, X)
# print("getting awf")
# awf = angular_wavefunction(theta, phi, l, m)

# print("getting psi")
# psi = rwf * awf

# psi = wavefunction(X, Y, Z, n, l, m)
psi = getpsi(X, Y, Z, psicode)

full_density = electron_density(psi)
mesh_threshold = mesh_threshold_factor * np.max(full_density)

maxd = np.max(full_density)
if maxd < mesh_threshold:
    print(f"max density less than threshold! {maxd} < {mesh_threshold}")
    sys.exit()

print(f"min max sum fd {np.min(full_density)} {np.max(full_density)} {np.sum(full_density)}")
# full_density /= np.sum(full_density)
# indices = full_density > threshold
# full_density[~indices] = 0
# print(f"min max sum fd {np.min(full_density)} {np.max(full_density)} {np.sum(full_density)}")

indices = full_density > threshold
radii = np.sqrt(X**2 + Y**2 + Z**2)
radii[~indices] = 0
max_r = 0.50 * np.max(radii)

while True:
    print("re-creating a grid ...")
    print(f"max_r is: {max_r}")

    x = np.linspace(-max_r, max_r, grid_size)
    y = np.linspace(-max_r, max_r, grid_size)
    z = np.linspace(-max_r, max_r, grid_size)
    rsize = max_r
    
    X, Y, Z = np.meshgrid(x, y, z)

    print("Re-calculating psi on new grid ...")
    psi = getpsi(X, Y, Z, psicode)

    full_density = electron_density(psi)
    mesh_threshold = mesh_threshold_factor * np.max(full_density)
    
    indices = full_density > threshold
    radii = np.sqrt(X**2 + Y**2 + Z**2)
    radii[~indices] = 0
    max_r = 1.10 * np.max(radii) # converge faster

    maxd = np.max(full_density)
    if maxd < mesh_threshold:
        print(f"max density less than threshold! {maxd} < {mesh_threshold}")
        sys.exit()
        
    if max_r < rsize:
        break

print("make pos stl ...")

stl_filename_pos = os.path.join(directory, fprefix + "pos.stl")
indices = psi > 0
density = np.copy(full_density)
density[~indices] = 0

maxd = np.max(density)
if maxd < mesh_threshold:
    print(f"max density less than threshold! {maxd} < {mesh_threshold}")
    sys.exit()

verts, faces = generate_mesh(density, mesh_threshold)
verts -= max_r
verts *= stl_factor

x_vals, y_vals, z_vals = verts[:, 0], verts[:, 1], verts[:, 2]
i_vals, j_vals, k_vals = faces[:, 0], faces[:, 1], faces[:, 2]

fig = go.Figure()
fig.add_trace(go.Mesh3d(x=x_vals, y=y_vals, z=z_vals, i=i_vals, j=j_vals, k=k_vals, color='red', opacity=0.5, name='Positive'))

# Convert to STL
stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        stl_mesh.vectors[i][j] = verts[f[j], :]

# Save the STL file
stl_mesh.save(stl_filename_pos)
print(f"STL file saved as {stl_filename_pos}")

# import trimesh

# print("Checking MESH")
# # Load the STL file
# tmesh = trimesh.load(stl_filename_pos)

# # Check if the mesh is watertight
# if not tmesh.is_watertight:
    # print("Mesh is not watertight. Attempting repair...")
    # # Attempt to fix non-watertight mesh
    # trimesh.repair.fix_inversion(tmesh)
    # trimesh.repair.fix_normals(tmesh)
    # trimesh.repair.fill_holes(tmesh)

    # if tmesh.is_watertight:
        # print("Mesh repair successful.")
        # # Export the repaired mesh
        # print(f"Re-saving {stl_filename_pos}")
        # tmesh.export(stl_filename_pos)
    # else:
        # print("Mesh repair failed.")
# else:
    # print(f"{stl_filename_pos} is watertight")

# second do negatives
if psicode != "psi100":
    print("make neg stl ...")
    stl_filename_neg = os.path.join(directory, fprefix + "neg.stl")
    indices = psi < 0
    density = np.copy(full_density)
    density[~indices] = 0
    
    verts, faces = generate_mesh(density, mesh_threshold)
    verts -= max_r
    verts *= stl_factor
    r_value_neg = np.max(np.abs(verts))

    x_vals, y_vals, z_vals = verts[:, 0], verts[:, 1], verts[:, 2]
    i_vals, j_vals, k_vals = faces[:, 0], faces[:, 1], faces[:, 2]

    fig.add_trace(go.Mesh3d(x=x_vals, y=y_vals, z=z_vals, i=i_vals, j=j_vals, k=k_vals, color='blue', opacity=0.5, name='Negative'))

# Convert to STL
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[f[j], :]

# Save the STL file
    stl_mesh.save(stl_filename_neg)
    print(f"STL file saved as {stl_filename_neg}")
    
    # print("Checking MESH")
    # # Load the STL file
    # tmesh = trimesh.load(stl_filename_neg)

    # # Check if the mesh is watertight
    # if not tmesh.is_watertight:
        # print("Mesh is not watertight. Attempting repair...")
        # # Attempt to fix non-watertight mesh
        # trimesh.repair.fix_inversion(tmesh)
        # trimesh.repair.fix_normals(tmesh)
        # trimesh.repair.fill_holes(tmesh)

        # if tmesh.is_watertight:
            # print("Mesh repair successful.")
            # # Export the repaired mesh
            # print(f"Re-saving {stl_filename_neg}")
            # tmesh.export(stl_filename_neg)
        # else:
            # print("Mesh repair failed.") 
    # else:
        # print(f"{stl_filename_neg} is watertight")
# end neg

# prompt: make 3 cylinders 100 mm long and 3 mm diameter that intersect at their centers and are all orthogonal to each other. Put this together as one shape and save it as an stl

print("make cylinders ...")
# Define cylinder parameters
# length = 2.0* r_value + 10.0  # mm
length = 2.0 * max_r  + 40.0  # mm
print(f"length is: {length}")
diameter = 6  # mm
radius = diameter / 2

# Create three cylinders intersecting at their centers and orthogonal to each other
def create_cylinder(length, radius, axis):
    #Creates a cylinder along a specified axis.
    num_segments = 50  # Adjust for desired resolution
    theta = np.linspace(0, 2 * np.pi, num_segments)
    z = np.linspace(-length / 2, length / 2, 2)

    # Create the cylinder mesh
    vertices = []
    faces = []
    for i in range(len(theta) - 1):
        for j in range(len(z) - 1):
            v1 = np.array([radius * np.cos(theta[i]), radius * np.sin(theta[i]), z[j]])
            v2 = np.array([radius * np.cos(theta[i + 1]), radius * np.sin(theta[i + 1]), z[j]])
            v3 = np.array([radius * np.cos(theta[i + 1]), radius * np.sin(theta[i + 1]), z[j + 1]])
            v4 = np.array([radius * np.cos(theta[i]), radius * np.sin(theta[i]), z[j + 1]])

            # Rotate the cylinder to align with the specified axis
            if axis == 'x':
                rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            elif axis == 'y':
                rotation_matrix = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            else:  # axis == 'z'
                rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            v1 = np.dot(rotation_matrix, v1)
            v2 = np.dot(rotation_matrix, v2)
            v3 = np.dot(rotation_matrix, v3)
            v4 = np.dot(rotation_matrix, v4)

            vertices.extend([v1, v2, v3, v4])
            faces.extend([[len(vertices) - 4, len(vertices) - 3, len(vertices) - 2],
                          [len(vertices) - 4, len(vertices) - 2, len(vertices) - 1]])

    return np.array(vertices), np.array(faces)


# Create three cylinders
vertices_x, faces_x = create_cylinder(length, radius, 'x')
vertices_y, faces_y = create_cylinder(length, radius, 'y')
vertices_z, faces_z = create_cylinder(length, radius, 'z')

# Combine the cylinders into a single mesh
all_vertices = np.concatenate((vertices_x, vertices_y, vertices_z))
all_faces = np.concatenate((faces_x, faces_y + len(vertices_x), faces_z + len(vertices_x) + len(vertices_y)))

print("make axes stl ...")
# Create STL mesh
stl_mesh = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(all_faces):
    for j in range(3):
        stl_mesh.vectors[i][j] = all_vertices[f[j], :]

# Save the STL file
stl_filename_axes = os.path.join(directory, fprefix + "axes.stl")
stl_mesh.save(stl_filename_axes)
print(f"STL file saved as {stl_filename_axes}")

# print("Checking MESH")
# # Load the STL file
# tmesh = trimesh.load(stl_filename_axes)

# # Check if the mesh is watertight
# if not tmesh.is_watertight:
    # print("Mesh is not watertight. Attempting repair...")
    # # Attempt to fix non-watertight mesh
    # trimesh.repair.fix_inversion(tmesh)
    # trimesh.repair.fix_normals(tmesh)
    # trimesh.repair.fill_holes(tmesh)

    # if tmesh.is_watertight:
        # print("Mesh repair successful.")
        # # Export the repaired mesh
        # print(f"Re-saving {stl_filename_axes}")
        # tmesh.export(stl_filename_axes)
    # else:
        # print("Mesh repair failed.")
# else:
    # print(f"{stl_filename_axes} is watertight")

# 3D plot using plotly

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title='X (Bohr radius)',
        yaxis_title='Y (Bohr radius)',
        zaxis_title='Z (Bohr radius)'
    ),
    title="3D Probability Surface",
    margin=dict(l=0, r=0, t=30, b=0),
)

# Show the plot
fig.show()





