"""
This code processes the data from data_plots_afespposter.txt to produce plots for the poster for the AFESP conference on Nov 4, 2024.
Date: 05-12-2024 (using RK2QC data from 05-12-2024)
Author: Amber te Winkel
"""

import numpy as np
import matplotlib.pyplot as plt
from data import *

def xval(nx):
    return np.linspace(0.,1.,nx, endpoint=False)

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

rmse04 = np.array([rmse04_orig[0], rmse04_orig[2], rmse04_orig[1]])
rmse125 = np.array([rmse125_orig[0], rmse125_orig[2], rmse125_orig[1]])
rmse20 = np.array([rmse20_orig[0], rmse20_orig[2], rmse20_orig[1]])
rmse25 = np.array([rmse25_orig[0], rmse25_orig[2], rmse25_orig[1]])

nx = [40,125,200,250]

#clrs = ['#762a83', '#af8dc3', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#7fbf7b', '#1b7837']
#clrs = ['#810f7c', '#8856a7', '#8c96c6']
#clrs = ['#e66101','#fdb863','#b2abd2','#5e3c99']

#clrs = ['#e66101','#b2abd2','#8c6bb1','#5e3c99']

clrs = ['#8c6bb1','#feb24c','#fc4e2a','#b10026']

fig1, ax1 = plt.subplots(figsize=(7,3))
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.18,
                 box.width, box.height * 0.95])
ax1.plot(xval(nx[-1]), analytic_1, label='Analytic', color='k', marker='', linestyle='-')
ax1.plot(xval(nx[0]), psi04_1, label='$C$ = 0.4', color=clrs[0], marker='', linestyle='-')
ax1.plot(xval(nx[1]), psi125_1, label='$C$ = 1.25', color=clrs[1], marker='', linestyle='-')
ax1.plot(xval(nx[2]), psi20_1, label='$C$ = 2.0', color=clrs[2], marker='', linestyle='--')
ax1.plot(xval(nx[-1]), psi25_1, label='$C$ = 2.5', color=clrs[3], marker='', linestyle=':')
ax1.set_xlabel('x')
ax1.set_ylabel('$\\Psi$')
ax1.set_xlim(0., 1.)
ax1.set_ylim(-0.35, 1.35)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
          ncol=7, fancybox=False, shadow=False, columnspacing=0.2)
#plt.savefig('./field1.jpg')
plt.savefig('./field1.png', format="png", dpi=1200)




fig3, ax3 = plt.subplots(figsize=(7,3))
box = ax3.get_position()
ax3.set_position([box.x0, box.y0 + box.height * 0.18,
                 box.width, box.height * 0.95])
ax3.plot(xval(nx[-1]), analytic_100, label='Analytic', color='k', marker='', linestyle='-')
ax3.plot(xval(nx[0]), psi04_100, label='$C$ = 0.4', color=clrs[0], marker='', linestyle='-')
ax3.plot(xval(nx[1]), psi125_100, label='$C$ = 1.25', color=clrs[1], marker='', linestyle='-')
ax3.plot(xval(nx[2]), psi20_100, label='$C$ = 2.0', color=clrs[2], marker='', linestyle='--')
ax3.plot(xval(nx[-1]), psi25_100, label='$C$ = 2.5', color=clrs[3], marker='', linestyle=':')
ax3.set_xlabel('x')
ax3.set_ylabel('$\\Psi$')
ax3.set_xlim(0., 1.)
ax3.set_ylim(-0.35, 1.35)
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
          ncol=7, fancybox=False, shadow=False, columnspacing=0.2)
#plt.savefig('./field100.jpg')
plt.savefig('./field100.png', format="png", dpi=1200)




dx = [0.005,0.01,0.02]#[0.0125, 0.025, 0.05] #06-12-2024: now actually dt
factor = 2
gridscale = np.logspace(0, np.log10(2*factor), num=10)
gridsizes = dx[0]*gridscale
firstorder = rmse04[0]*gridscale
secondorder = rmse04[0]*gridscale*gridscale
thirdorder = rmse04[0]*gridscale*gridscale*gridscale

fig2, ax2 = plt.subplots(figsize=(6,3))
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.18,
                 box.width, box.height * 0.95])
ax2.plot(gridsizes, firstorder, color='lightgrey', linestyle='-')
ax2.plot(gridsizes, secondorder, color='lightgrey', linestyle='-')
ax2.plot(gridsizes, thirdorder, color='lightgrey', linestyle='-')
ax2.plot(dx, rmse04, label='$C$ = 0.4', color=clrs[0], marker='x')
ax2.plot(dx, rmse125, label='$C$ = 1.25', color=clrs[1], marker='x')
ax2.plot(dx, rmse20, label='$C$ = 2.0', color=clrs[2], marker='+')
ax2.plot(dx, rmse25, label='$C$ = 2.5', color=clrs[3], marker='+')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('dt')
ax2.set_ylabel('RMSE')
#ax2.set_xticks([0.005,0.01,0.02])
#ax2.set_xticklabels(['5 x 10$^{-3}$', '10$^{-2}$', '2 x 10$^{-2}$'])
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
          ncol=6, fancybox=False, shadow=False, columnspacing=0.2)
plt.savefig('./rmse.png', format="png", dpi=1200)
#print(f"Default Matplotlib DPI: {plt.rcParams['figure.dpi']}")

