
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import heapq

from main_functions import *


def calculate_field_lines(data, L, num_points, seeding, method, h=0.1):    
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
            curve_length = np.sum(np.linalg.norm(sline[1:] - sline[:-1], axis=1))
            if curve_length > 0.1:
                i += 1
                if color is not None:
                    axis.plot(sline[:,1], sline[:,0], linewidth=0.5, color=color, alpha=0.3)
                    axis.plot(sline[-1,1], sline[-1,0], "*", color=color) 
                else:
                    axis.plot(sline[:,1], sline[:,0], linewidth=0.5)

def plot_seeding_points(axis, data, points):
    X, Y, _ = data.shape
    
    print("shape: ", points.shape) 

    xs = np.linspace(0, X-1, X)
    ys = np.linspace(0, Y-1, Y)
    
    axis.contour(ys, xs, np.linalg.norm(data, axis=2), [0], linewidths=0.3)
    
    axis.set_xlim(0, X)
    axis.set_ylim(Y, 0)
    
    axis.scatter(points[:,0], points[:,1], 20/np.sqrt(len(points)))

def within_range(field_lines, X, Y):
    s_x_min = field_lines[:, :, :, 0] >= 0
    s_x_max = field_lines[:, :, :, 0] < X
    
    s_y_min = field_lines[:, :, :, 1] >= 0
    s_y_max = field_lines[:, :, :, 1] < Y
    
    s_x = np.logical_and(s_x_min, s_x_max)
    s_y = np.logical_and(s_y_min, s_y_max)

    return np.logical_and(s_x, s_y)


def plot_seeding_vs_length(data1, data2):
    labels = ["Uniform", "Random", "Density based", "Feature based"]

    for (data, data_label) in zip([data1, data2], ["isabel", "metsim"]):
        fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
         
        X, Y, _ = data.shape

        for (i, seeding_strategy) in enumerate([feature_based]): #uniform, random, density_based, feature_based]):
            N = [2*10, 2*40, 2*160, 2*640]

            for j in range(4):
                length = N[3-j]
                num_points = N[j]
                print(length, num_points)
                
                points, field_lines = calculate_field_lines(data, length, num_points, seeding_strategy, "RK4")
                """
                for axes in [axes1, axes2]:
                    axes[i][j].set_xticks([])
                    axes[i][j].set_yticks([])
                    axes[-1][j].set_xlabel(f"N: {length}")
                    axes[i][0].set_ylabel(labels[i])
                """
                print(points)
                plot_field_lines(axes1[i][j], data, field_lines)
                plot_seeding_points(axes2[i][j], data, points)
                plt.show()
                exit()
        fig1.savefig(f"{data_label}_field_lines.png", dpi=300)
        fig2.savefig(f"{data_label}_points.png", dpi=300)


def plot_FE_vs_RK4_field_lines(data):
    length = 10000
    num_points = 2

    fig, axes = plt.subplots(2, 2, figsize=(6, 5))

    reds = ["gold", "tab:orange", "tab:red", "maroon", "tab:brown"]

    hvalues = [10, 1, 0.1, 0.01, 0.001]
    lines = []

    field_lines_FE = []
    field_lines_RK4 = []

    for h in hvalues:
        L = int(length/h)
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



file1 = "data/isabel_2d.h5"
file2 = "data/metsim1_2d.h5"

data1 = read_data(file1)
data2 = read_data(file2)

#normalize_data(data1)
#normalize_data(data2)

plot_seeding_vs_length(data1, data2)
