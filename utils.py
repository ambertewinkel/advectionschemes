# Module file for utilities functions used in main.py, schemes.py and experiments.py
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from pathlib import Path
import sys
import xarray as xr

def design_figure(filename, title, xlabel, ylabel, bool_ylim = False, ylim1=0.0, ylim2=0.0):
    if bool_ylim == True: plt.ylim(ylim1, ylim2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

def to_vector(array, length):
    """
    This function checks whether an array is 0D (scalar) or 1D.
    If scalar, it outputs a vector of len=length filled with the scalar values.
    If vector, check whether it has len=length.
    Note: this assumes array is a numpy array.
    --- Input --- 
    array   : np.array of any dimension
    length  : scalar with length of 1D vector that we want outputted
    --- Output ---
    If no ValueError produced, this function outputs
    res     : 1D np.array of dimension length. If input array was scalar, output array 
            is filled with these scalar values.
    """

    if np.isscalar(array):
        res = np.full(length, array)
    elif array.ndim == 1:
        if len(array) != length:
            print('Expected length:', length)
            print('Array length:', len(array))
            raise ValueError('Array input (vector) in scalar2vector does not have the expected length.')
        else:
            res = array.copy()
    else:
        raise TypeError('Array input in scalar2vector is neither a scalar nor vector.')
    
    return res

def plot_Courant(x, c):
    plt.plot(x, c)
    plt.axhline(1.0, color='grey', linestyle=':')
    plt.axvline(1.0)
    plt.title('Courant number')
    plt.xlabel('x')
    plt.ylabel('C')
    plt.tight_layout()
    plt.savefig('Courant.pdf')
    plt.clf()

def plot_grid(x, dx):
    plt.plot(x)
    plt.title('Stretching: x against i')
    plt.xlabel('i')
    plt.ylabel('x')
    plt.tight_layout()
    plt.savefig('gridpoints.pdf')
    plt.clf()
    plt.plot(dx)
    plt.xlabel('i')
    plt.ylabel('dx')
    plt.title('Stretching: dx against i')
    plt.tight_layout()
    plt.savefig('gridspacing.pdf')
    plt.clf()

def make_animation(init, fn, ntmax, filebasename, nt, dt, uf, dxc, niter=1):
    # Script to create animation from set of pdf files, based on create_gif.py from FVM/PMAP data analysis

    # Directory to put snapshots in
    dirname = './animations/' + filebasename +'/'
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        print("Folder %s created!" % dirname)
    else:
        print("Folder %s already exists" % dirname)

    # Calculating field snapshots to plot
    for it in range(len(ntmax)):
        if 'Jacobi' in fn or 'GaussSeidel' in fn:
            locals()[f'psi1_{fn}'] = fn(init.copy(), it, dt, uf, dxc, niter)
            locals()[f'psi2_{fn}'] = fn(init.copy(), it, dt, uf, dxc, niter)
        else:
            locals()[f'psi1_{fn}'] = fn(init.copy(), it, dt, uf, dxc)
            locals()[f'psi2_{fn}'] = fn(init.copy(), it, dt, uf, dxc)
        
    # Plotting and saving snapshots
            


    # Creating animation from snapshots
            
            
    ######


    filenames = []

    plotdir = dirname + 'plots/'
    output = ['tracer', 'u', 'v', 'w']
    i = 0
    while f'u_data_{i}.png' in os.listdir(plotdir):
        for o in output:
            file = f'{o}_data_{i}.png'
            filenames.append(file)
        i += 1

    images_trc = []
    images_u = []
    images_v = []
    images_w = []
    for filename in filenames:
        if 'tracer_data_' in filename:
            images_trc.append(imageio.imread(plotdir + filename))
        elif 'u_data_' in filename:
            images_u.append(imageio.imread(plotdir + filename))
        elif 'v_data_' in filename:
            images_v.append(imageio.imread(plotdir + filename))
        elif 'w_data_' in filename:
            images_w.append(imageio.imread(plotdir + filename))
        else: 
            print('Filename', filename, 'could not be assigned to an animation.')
    imageio.mimsave(gifdir + '/tracer.gif', images_trc)
    imageio.mimsave(gifdir + '/u.gif', images_u)
    imageio.mimsave(gifdir + '/v.gif', images_v)
    imageio.mimsave(gifdir + '/w.gif', images_w)