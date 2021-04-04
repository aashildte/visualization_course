
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

def read_data(filename):

    d1 = h5py.File(filename, 'r')

    xs = np.array(d1["Velocity"]["X-comp"])
    ys = np.array(d1["Velocity"]["Y-comp"])

    data = np.zeros((xs.shape[0], xs.shape[0], 2))

    data[:, :, 0] = xs
    data[:, :, 1] = ys

    return data

def interpolate_2d(xs, ys, data):
    xcomp = RectBivariateSpline(xs, ys, data[:,:,0])
    ycomp = RectBivariateSpline(xs, ys, data[:,:,1])

    f = lambda x, y: np.array((xcomp(x, y, grid=False), ycomp(x, y, grid=False))).transpose()
    return f


def calculate_streamlines(f, method, midpoints, length, h=0.1):
    L = 2*int(length/(2*h)) + 1

    streamline = np.zeros((len(midpoints), 2*L + 1, 2))

    cur_pos = midpoints[:]

    streamline[:,L] = cur_pos
    index = L - 1

    for ch in [-1, 1]:
        for i in range(L):
            cur_x = cur_pos[:,0]
            cur_y = cur_pos[:,1]
            if method=="RK4":
                k1 = ch*f(cur_x, cur_y)
                k2 = k3 = ch*f(cur_x + h/2, cur_y + h/2)
                k4 = ch*f(cur_x + h, cur_y + h)
                
                cur_pos = cur_pos + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
            else:
                cur_pos = cur_pos + ch*h*f(cur_x, cur_y)
            
            streamline[:,index] = cur_pos
            index += ch
        
        cur_pos = midpoints[:]
        index = L + 1

    return streamline


def field_lines(data, L, num_points, seeding, method):
    
    X, Y, _ = data.shape

    xs = np.linspace(0, X-1, X)
    ys = np.linspace(0, Y-1, Y)
    
    fun = interpolate_2d(xs, ys, data)

    if seeding=="random":
        xs = X*np.random.rand(num_points)
        ys = Y*np.random.rand(num_points)

        positions = np.zeros((num_points, 2))
        positions[:,0] = xs
        positions[:,1] = ys

    slines = calculate_streamlines(fun, "RK4", positions, L)

    for sline in slines:
        p = plt.plot(sline[:,0], sline[:,1])
        c = p[0].get_color() 
        plt.plot(sline[L//2,0], sline[L//2,1], ">", markersize=2, color=c) 

    plt.show()

def within_range(streamlines, X, Y):
    s_x_min = streamlines[:, :, :, 0] >= 0
    s_x_max = streamlines[:, :, :, 0] < X
    
    s_y_min = streamlines[:, :, :, 1] >= 0
    s_y_max = streamlines[:, :, :, 1] < Y
    
    s_x = np.logical_and(s_x_min, s_x_max)
    s_y = np.logical_and(s_y_min, s_y_max)

    return np.logical_and(s_x, s_y)


def lic(data, length, method):

    X, Y, _ = data.shape
    blur = np.random.uniform(size=(X, Y))
    new_im = np.zeros_like(blur)
    
    xs = np.linspace(0, X-1, X)
    ys = np.linspace(0, Y-1, Y)
     
    # normalize directions
    norm = np.linalg.norm(data, axis=2)

    np.divide(data[:,:,0], norm, out=data[:,:,0], where=norm>1E-14)
    np.divide(data[:,:,1], norm, out=data[:,:,1], where=norm>1E-14)

    fun = interpolate_2d(xs, ys, data)

    Ys, Xs = np.meshgrid(xs, ys)

    xs2 = Xs.flatten()
    ys2 = Ys.flatten()

    positions = np.zeros((len(xs2), 2))
    positions[:,0] = xs2
    positions[:,1] = ys2

    # calculate streamlines

    streamlines = calculate_streamlines(fun, method, positions, length, h=0.5)
    _, L, _ = streamlines.shape
    streamlines = np.reshape(streamlines, (X, Y, L, 2)).astype(int)
    
    wr = within_range(streamlines, X, Y)     # filter for out of bounds errors

    streamlines[:, :, :, 0] = np.where(wr, streamlines[:, :, :, 0], 0)
    streamlines[:, :, :, 1] = np.where(wr, streamlines[:, :, :, 1], 0)

    new_im = np.average(blur[streamlines[:, :, :, 0], streamlines[:, :, :, 1]], axis=2, weights=wr)

    new_im = np.where(norm > 1E-10, new_im, 0)   # make initial zero values zero

    plt.figure()
    plt.imshow(new_im)
    plt.show()

file1 = "data/isabel_2d.h5"
file2 = "data/metsim1_2d.h5"

data1 = read_data(file1)
data2 = read_data(file2)

#field_lines(data1, 5, 1000, "random", "RK4")
lic(data1, 50, "RK4")
