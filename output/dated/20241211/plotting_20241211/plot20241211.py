"""
This code processes the data from data_plots_afespposter.txt to produce plots for the poster for the AFESP conference on Nov 4, 2024.
Date: 11-12-2024 (using RK2QC data from 11-12-2024)
Author: Amber te Winkel
--> PLOT PRODUCED FOR MC3 REPORT
NOTE that the x coordinates are calculated differently here than before. I was plotting xf before and now I am plotting xc.
"""

import numpy as np
import matplotlib.pyplot as plt
from data import *

def xval(nx):
    #return np.linspace(0.,1.,nx, endpoint=False)
    xf, dxc, xc, dxf = coords_uniform(1., nx)
    return xc

def coords_uniform(xmax, imax):
    """
    This function implements a uniform grid spacing.
    We assume a periodic domain that ranges from 0 to xmax in size. xmax is not included in x-array
    --- Input:
    xmax    : float, domain size
    imax    : int, number of grid points
    --- Output:
    xf       : array of floats, spatial points of cell faces
    dxc      : array of floats, grid spacing between cell faces (dxc[i] = xf[i+1] - xf[i]), i.e., grid box size
    xc       : array of floats, spatial points of cell centers (xc[i+1] = 0.5*(xf[i+1] + xf[i]))
    dxf      : array of floats, grid spacing between cell centers (dxf[i] = xc[i+1] - xc[i])
    """
    # Initialisation
    xf, dxc, xc, dxf = np.zeros(imax), np.zeros(imax), np.zeros(imax), np.zeros(imax)
    
    # Setting grid values
    dxc = np.full(imax, float(xmax/imax)) # define the grid spacing

    # Calculating other grid quantities
    for i in range(len(dxc)-1):
        xf[i+1] = xf[i] + dxc[i]
    xc = 0.5*(np.roll(xf,-1) + xf)
    xc[-1] = 0.5*(xmax + xf[-1]) # periodic
    dxf = dxc.copy()
   
    return  xf, dxc, xc, dxf

def design_figure(filename, title, xlabel, ylabel, xlim1, xlim2, bool_ylim = False, ylim1=0.0, ylim2=0.0):
    if bool_ylim == True: plt.ylim(ylim1, ylim2)
    plt.xlim(xlim1, xlim2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

nx = [40,125,200,250]
clrs = ['#8c6bb1','#feb24c','#fc4e2a','#b10026']

fig3, ax3 = plt.subplots(figsize=(7,3))
box = ax3.get_position()
ax3.set_position([box.x0, box.y0 + box.height * 0.18,
                 box.width, box.height * 0.95])
ax3.plot(xval(nx[-1]), analytic_100, label='Analytic', color='k', marker='', linestyle='-')
#ax3.plot(xval(nx[0]), analytic04_100, label='Analytic', color='gray', marker='', linestyle='-')
ax3.plot(xval(nx[0]), psi04_100, label='$C$ = 0.4', color=clrs[0], marker='', linestyle='-')
ax3.plot(xval(nx[1]), psi125_100, label='$C$ = 1.25', color=clrs[1], marker='', linestyle='-')
ax3.plot(xval(nx[2]), psi20_100, label='$C$ = 2.0', color=clrs[2], marker='', linestyle='--')
ax3.plot(xval(nx[-1]), psi25_100, label='$C$ = 2.5', color=clrs[3], marker='', linestyle=':')
ax3.set_xlabel('x')
ax3.set_ylabel('$\\Psi$')
ax3.set_xlim(0., 1.)
ax3.set_ylim(-0.1, 1.1)
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
          ncol=7, fancybox=False, shadow=False, columnspacing=0.2)
#plt.savefig('./field100.jpg')
plt.savefig('./field100.png', format="png", dpi=1200)