# Code for utilities functions used in main.py, schemes.py and experiments.py
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

import numpy as np
import matplotlib.pyplot as plt

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