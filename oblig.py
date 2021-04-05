
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import Delaunay
import heapq


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

"""
def curl(data):
    fx_dy = np.gradient(data[:,:,0], axis=1)
    fy_dx = np.gradient(data[:,:,1], axis=0)

    return fy_dx - fx_dy
"""

def curl(fun, x, y, h=1E-10):
    fx_dy = fun(x, y + h)[1]/h
    fy_dx = fun(x + h, y)[0]/h

    return fy_dx - fx_dy


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


def random(X, Y, num_points, fun, method, L):
    xs = X*np.random.rand(num_points)
    ys = Y*np.random.rand(num_points)
    
    positions = np.zeros((num_points, 2))
    positions[:,0] = xs
    positions[:,1] = ys

    return calculate_streamlines(fun, method, positions, L)

def uniform(X, Y, num_points, fun, method, L):
    N = int(np.sqrt(num_points))
    xs = X*np.linspace(0, 1, N)
    ys = Y*np.linspace(0, 1, N)

    ys, xs = np.meshgrid(xs, ys)
    xs = xs.flatten()
    ys = ys.flatten()

    num_points = N*N

    positions = np.zeros((num_points, 2))
    positions[:,0] = xs
    positions[:,1] = ys

    return calculate_streamlines(fun, method, positions, L)

def density_based(X, Y, num_points, fun, method, L):
    N = num_points
    xs = X*np.random.rand(N)
    ys = Y*np.random.rand(N)

    positions = np.zeros((N, 2))
    positions[:,0] = xs
    positions[:,1] = ys
    
    triangles = Delaunay(positions).simplices
    queue = []

    for t in triangles:
        coords = ((xs[t[0]], ys[t[0]]),
                  (xs[t[1]], ys[t[1]]),
                  (xs[t[2]], ys[t[2]]))
        
        midpoint = np.mean(np.array(coords), axis=0)
        heapq.heappush(queue, (0, tuple(midpoint), coords))
    
    next_queue = []
    streamlines = []
    threshold = 5

    # now let's get started

    while len(queue)>0 and len(streamlines) < num_points:
        new_point = heapq.heappop(queue)[1]
        #plt.plot([new_point[1]], [new_point[0]], '*')
        streamline = calculate_streamlines(fun, method, np.array([new_point]), L)[0]

        streamlines.append(np.array(streamline))

        for item in queue:
            weight = item[0]
            point = item[1]
            triangle = item[2]

            min_dist = min(np.linalg.norm(streamline - np.array(point), axis=1))

            if min_dist > threshold:
                heapq.heappush(next_queue, (weight+min_dist, point, triangle))

        queue = next_queue
        next_queue = []

    return streamlines


def feature_based(X, Y, num_points, fun, method, L):

    # calculate a number of random points; sort according to vorticity, use the
    # num_points with highest values to display the vector field
    xs = X*np.random.rand(int(num_points*2))
    ys = Y*np.random.rand(int(num_points*2))

    ranked = []

    for (x, y) in zip(xs, ys):
        c = curl(fun, x, y)

        ranked.append((c, x, y))

    ranked.sort()
    ranked = np.array(ranked)

    positions = np.zeros((num_points, 2))
    positions[:,0] = ranked[:num_points,1]
    positions[:,1] = ranked[:num_points,2]

    return calculate_streamlines(fun, method, positions, L)
    

def field_lines(data, L, num_points, seeding, method):
    
    X, Y, _ = data.shape

    xs = np.linspace(0, X-1, X)
    ys = np.linspace(0, Y-1, Y)
    
    fun = interpolate_2d(xs, ys, data)

    streamlines = seeding(X, Y, num_points, fun, method, L)

    for sline in streamlines:
        p = plt.plot(sline[:,1], sline[:,0], linewidth=0.5)
        #c = p[0].get_color()
        #plt.plot(sline[L//2,0], sline[L//2,1], ">", markersize=2, color=c) 
    
    plt.xlim(0, X)
    plt.ylim(Y, 0)
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

#field_lines(data1, 50, 10, "uniform", "RK4")
field_lines(data1, 1000, 100, feature_based, "RK4")
#lic(data1, 5, "RK4")

