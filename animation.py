# This file calculates data for and produces animations. It can be called from other files (make_animation(...) -- input is a single scheme and various timestepping, grid, and velocity info) but also executed itself. 
# Author:   Amber te Winkel
# Email:    a.j.tewinkel@pgr.reading.ac.uk


import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from pathlib import Path
import sys
import xarray as xr
import utils as ut
import grid as gr
import analytic as an
import schemes as sch


def make_animation(fn, filebasename, nt, dt, uf, dxc, xc, xmax, uc, niter=1):
    # Script to create animation from set of pdf files, based on create_gif.py from FVM/PMAP data analysis

    # Directory to put snapshots in
    dirname = './animations/' + filebasename +'/'
    plotdir = dirname + 'plots/'
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        print("Folder %s created!" % dirname)
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)
        print("Folder %s created!" % plotdir)

    # Calculate initial functions
    field_in = an.analytic1(xc, xmax)
    psi2_in = an.analytic2(xc, xmax)

    filenames1, images1, filenames2, images2 = [], [], [], []
    for it in range(nt+1):
        f = getattr(sch, f'{fn}')
        
        # Calculate analytic solutions
        analytic = an.analytic1(xc, xmax, uc, it*dt)
        psi2_an = an.analytic2(xc, xmax, uc, it*dt)
        
        # Calculating field snapshots to plot
        if 'Jacobi' in fn or 'GaussSeidel' in fn:
            locals()[f'psi1_{fn}'] = f(field_in.copy(), it, dt, uf, dxc, niter)[-1]
            locals()[f'psi2_{fn}'] = f(psi2_in.copy(), it, dt, uf, dxc, niter)[-1]
        else:
            locals()[f'psi1_{fn}'] = f(field_in.copy(), it, dt, uf, dxc)[-1]
            locals()[f'psi2_{fn}'] = f(psi2_in.copy(), it, dt, uf, dxc)[-1]
        
        # Plotting and saving snapshots
        # Initial condition 1
        plt.plot(xc, field_in, label='Initial', linestyle='-', color='grey')
        plt.plot(xc, analytic, label='Analytic', linestyle='-', color='k')
        plt.plot(xc, locals()[f'psi1_{fn}'], label=f'{fn}', marker='x', linestyle='-', color='blue')
        ut.design_figure(f'{plotdir}Psi1_{fn}_{it}.png', f'$\\Psi_1$ at t={it*dt:.2f}', \
                        'x', '$\\Psi_1$', True, -0.1, 1.1)
        
        # Initial condition 2
        plt.plot(xc, psi2_in, label='Initial', linestyle='-', color='grey')
        plt.plot(xc, psi2_an, label='Analytic', linestyle='-', color='k')
        plt.plot(xc, locals()[f'psi2_{fn}'], label=f'{fn}', marker='x', linestyle='-', color='blue')
        ut.design_figure(f'{plotdir}Psi2_{fn}_{it}.png', f'$\\Psi_2$ at t={it*dt:.2f}', \
                        'x', '$\\Psi_2$', True, -0.1, 1.1)
        
        filenames1.append(f'{plotdir}Psi1_{fn}_{it}.png')
        filenames2.append(f'{plotdir}Psi2_{fn}_{it}.png')

    # Creating animation from snapshots
    for filename in filenames1:
        images1.append(imageio.imread(filename))
    imageio.mimsave(f'{dirname}{filebasename}_1.gif', images1, duration=500)

    for filename in filenames2:
        images2.append(imageio.imread(filename))
    imageio.mimsave(f'{dirname}{filebasename}_2.gif', images2, duration=500)


def produce_standalone_animation():
    # Initial conditions
    dt = 0.1                    # time step
    nt = 100                    # number of time steps
    nx = 40                     # number of points in space
    xmax = 2.0                  # physical domain parameters
    uf = np.full(nx, 0.2)       # velocity at faces (assume constant)
    uc = np.full(nx, 0.2)       # velocity at centers

    keep_model_stable = False
    if keep_model_stable == True:
        cmax = 1.
        dxcmin = np.min(0.5*dt*(np.roll(uf,-1) + uf)/cmax)
    else:
        dxcmin = 0.
        
    xf, dxc, xc, dxf = gr.coords_centralstretching(xmax, nx, nx/2, dxcmin=dxcmin) # points in space, length of spatial step
    cc = 0.5*dt*(np.roll(uf,-1) + uf)/dxc # Courant number (defined at cell center)
    niter = 1                   # number of iterations (for Jacobi or Gauss-Seidel)
    
    make_animation('hybrid_MPDATA_BTBS1J', 'hybrid_MPDATA_BTBS1J_notkeptstable', nt, dt, uf, dxc, xc, xmax, uc)


def create_single_animation_from_data(filebasename, field, analytic, nt, dt, xc, animdir):
    """This function creates an animation from a given data file of a single scheme. The input is a 2D field of a single scheme (shape = 1d time x 1d space), analytic solution (shape = 1d time x 1d space), nt, dx in the centers (dxc; shape 1d space) and ....
    This is a function that is to be called from other files, not from produce_standalone_animation()."""

    # Directory to put animation and subdirectory for plots in
    plotdir = animdir + 'plots_' + filebasename + '/'
    os.mkdir(plotdir)

    # Calculate initial functions
    field_in = analytic[0]

    # Timestepping loop to create plots and save filenames
    filenames, images = [], []
    for it in range(nt+1):              
        # Plot each timestep in a figure and save in the plots subdirectory
        plt.plot(xc, field_in, label='Initial', linestyle='-', color='grey')
        plt.plot(xc, analytic[it], label='Analytic', linestyle='-', color='k')
        plt.plot(xc, field[it], label='Numerical', marker='x', linestyle='-', color='blue')
        ut.design_figure(f'{plotdir}field_{it}.png', '', f'{filebasename} at t={it*dt:.2f}', \
                        'x', '$field$', True, -0.5, 1.5)
        filenames.append(f'{plotdir}field_{it}.png')

    # Create animation from plots in the plots subdirectory
    for filename in filenames:
        images.append(imageio.imread(filename))
    anim_filename = f'{animdir}{filebasename}.gif'
    imageio.mimsave(anim_filename, images, duration=500)

    # Remove .png files used to create the animation
    for filename in filenames:
        os.remove(filename)
    os.rmdir(plotdir)


def create_animation_from_data(fields, nfields, analytic, nt, dt, xc, animdir, plot_args):
    """This function creates an animation from a given data file of a single scheme. The input is a 2D field of a single scheme (shape = 1d time x 1d space), analytic solution (shape = 1d time x 1d space), nt, dx in the centers (dxc; shape 1d space) and ....
    This is a function that is to be called from other files, not from produce_standalone_animation()."""

    # Directory to put animation and subdirectory for plots in
    plotdir = animdir + 'plots/'
    os.mkdir(plotdir)

    # Calculate initial functions
    field_in = analytic[0]

    # Timestepping loop to create plots and save filenames
    filenames, images = [], []
    for it in range(nt+1):   
        # Plot each timestep in a figure and save in the plots subdirectory
        plt.plot(xc, field_in, label='Initial', linestyle='-', color='grey')
        plt.plot(xc, analytic[it], label='Analytic', linestyle='-', color='k')
        for si in range(nfields):   
            field = fields[si]        
            plt.plot(xc, field[it], **plot_args[si])
        ut.design_figure(f'{plotdir}fields_{it}.png', '', f'Psi at t={it*dt:.2f}', \
                        'x', 'Psi', True, -0.5, 1.5)
        filenames.append(f'{plotdir}fields_{it}.png')

    # Create animation from plots in the plots subdirectory
    for filename in filenames:
        images.append(imageio.imread(filename))
    anim_filename = f'{animdir}animation.gif'
    imageio.mimsave(anim_filename, images, duration=500)

    # Remove .png files used to create the animation
    for filename in filenames:
        os.remove(filename)
    os.rmdir(plotdir)


if __name__ == "__main__": produce_standalone_animation()