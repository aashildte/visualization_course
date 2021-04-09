
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import heapq

from main_functions import *


def plot_FE_vs_RK4_lic(data):
    length = 40
    X, Y, _ = data.shape
    blur = np.random.uniform(size=(X, Y))
    
    
    hvalues = [5, 1, 0.5]

    fig, axes = plt.subplots(len(hvalues), 2, figsize=(6, 5))

    for (i, h) in enumerate(hvalues):
        for (j, method) in enumerate(["FE", "RK4"]):
            axis = axes[i][j]
            lic_im = lic(axis, data, length, method, blur, h)

            axis.set_xlim(255, 310)
            axis.set_ylim(405, 355)
            axis.set_xticks([])
            axis.set_yticks([])

        axes[i][0].set_ylabel(f"h : {h}")

    axes[0][0].set_title("Forward Euler")
    axes[0][1].set_title("Runge-Kutta 4")

    plt.savefig("fe_vs_rk4_lic.png", dpi=300)
    plt.show()


def plot_different_lengths_lic(data):
    X, Y, _ = data.shape

    lengths = [X//100, X//50, X//10, X//5, X//2]

    fig, axes = plt.subplots(1, len(lengths), figsize=(6, 5))
    
    for i in range(1):
        blur = np.random.uniform(size=((i+1)*X, (i+1)*Y))

        for (j, l) in enumerate(lengths):
            axis = axes[i]
            im = lic(data, l, "RK4", blur, 1)

            axis.imshow(im)

            axis.set_xticks([])
            axis.set_yticks([])
            #plt.show()
            #exit()

    axes[0][0].set_title(lengths[0])
    axes[0][1].set_title(lengths[1])
    axes[0][2].set_title(lengths[2])

    plt.savefig("l_vs_h_lic.png", dpi=300)
    plt.show()

file1 = "data/isabel_2d.h5"
file2 = "data/metsim1_2d.h5"

data1 = read_data(file1)
data2 = read_data(file2)

normalize_data(data1)
normalize_data(data2)

plot_different_lengths_lic(data1)
#plot_FE_vs_RK4_lic(data1)
