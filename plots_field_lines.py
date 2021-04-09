
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import heapq

from main_functions import *


def calculate_field_lines(data, L, num_points, seeding, method="RK4", h=0.1):
    X, Y, _ = data.shape

    xs = np.linspace(0, X-1, X)
    ys = np.linspace(0, Y-1, Y)
    
    fun = interpolate_2d(xs, ys, data)
    
    return seeding(X, Y, num_points, fun, method, L, h)

def plot_field_lines(axis, data, field_lines, color=None):
    X, Y, _ = data.shape
    
    xs = np.linspace(0, X-1, X)
    ys = np.linspace(0, Y-1, Y)
    
    axis.contour(ys, xs, np.linalg.norm(data, axis=2), [0], linewidths=0.3)
    
    axis.set_xlim(0, X)
    axis.set_ylim(Y, 0)
    i = 0
    for sline in field_lines:
        if len(sline) > 3:
            i += 1
            
            if color is not None:
                axis.plot(sline[:,1], sline[:,0], linewidth=0.5, color=color, alpha=0.3)
                axis.plot(sline[-1,1], sline[-1,0], "*", color=color) 
            else:
                axis.plot(sline[:,1], sline[:,0], linewidth=0.8)

def plot_seeding_points(axis, data, points):
    X, Y, _ = data.shape
    
    print("shape: ", points.shape) 

    xs = np.linspace(0, X-1, X)
    ys = np.linspace(0, Y-1, Y)
    
    axis.contour(ys, xs, np.linalg.norm(data, axis=2), [0], linewidths=0.3)
    
    axis.set_xlim(0, X)
    axis.set_ylim(Y, 0)
    
    axis.scatter(points[:,1], points[:,0], 20/np.sqrt(len(points)))



def plot_seeding_vs_length(data1, data2):
    labels = ["Uniform", "Random", "Density based"]

    N = [10, 50, 100, 500, 1000]

    for (data, data_label) in zip([data1, data2], ["isabel", "metsim"]):
        fig1, axes1 = plt.subplots(len(N), 3, figsize=(15, 25))
        fig2, axes2 = plt.subplots(len(N), 3, figsize=(15, 25))
         
        X, Y, _ = data.shape

        for i in range(len(N)):
            for (j, seeding_strategy) in enumerate([uniform, random, density_based]):
                num_points = N[i]
                length = (X/500)*N[len(N) - 1 - i]
                
                print(i, j, axes1.shape, axes2.shape) 

                points, field_lines = calculate_field_lines(data, length, num_points, seeding_strategy)
                
                for axes in [axes1, axes2]:
                    axes[i][j].set_xticks([])
                    axes[i][j].set_yticks([])
                    axes[i][0].set_ylabel(f"N: {num_points}")
                    axes[0][j].set_title(labels[j])
                
                plot_field_lines(axes1[i][j], data, field_lines)
                plot_seeding_points(axes2[i][j], data, points)
                
        fig1.savefig(f"{data_label}_field_lines.png", dpi=300)
        fig2.savefig(f"{data_label}_points.png", dpi=300)

def plot_FE_vs_RK4_field_lines(data):
    length = 1000
    num_points = 1

    fig, axes = plt.subplots(2, 2, figsize=(6, 5))

    reds = ["yellow", "gold", "tab:orange", "tab:red", "firebrick", "black"]

    hvalues = [5, 1, 0.5, 0.01, 0.005] #, 0.001]
    lines = []
    
    field_lines_FE = []
    field_lines_RK4 = []

    for h in hvalues:
        L = int(2*length/(h))
        print("h: ", h)
        _, fl = calculate_field_lines(data, L, num_points, corner, "FE", h)
        field_lines_FE.append(fl[0])
        _, fl = calculate_field_lines(data, L, num_points, corner, "RK4", h)
        field_lines_RK4.append(fl[0])
    
    for i in range(2):
        for (field_lines, axis) in zip([field_lines_FE, field_lines_RK4], axes[i]):
            for (c, h, line) in zip(reds, hvalues, field_lines):
                plot_field_lines(axis, data, [line], color=c)
                axis.set_xticks([])
                axis.set_yticks([])
                
    axes[0][0].set_title("Forward Euler")
    axes[0][1].set_title("Runge-Kutta 4")
    
    axes[1][0].set_xlim(255, 310)
    axes[1][0].set_ylim(405, 355)
    axes[1][1].set_xlim(255, 310)
    axes[1][1].set_ylim(405, 355)
    
    for (h, c) in zip(hvalues, reds):
        axes[0][1].plot([0], [0], color=c, label=f"h: {h}")

    axes[0][1].legend(bbox_to_anchor=[1.5, 1])

    plt.tight_layout()
    plt.savefig("fe_vs_rk4_field_lines.png", dpi=300)
    plt.show()

file1 = "data/isabel_2d.h5"
file2 = "data/metsim1_2d.h5"

data1 = read_data(file1)
data2 = read_data(file2)

normalize_data(data1)
normalize_data(data2)

plot_seeding_vs_length(data1, data2)
#plot_FE_vs_RK4_field_lines(data1)
#plot_FE_vs_RK4_field_lines(data2)

