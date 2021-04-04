
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


def calculate_streamlines(f, method, midpoints, L):
    streamline = np.zeros((len(midpoints), 2*L + 1, 2))

    h = 0.5

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
        plt.plot(slines[x,:,0][::(L//2)], slines[x,:,1][::(L//2)], ">", markersize=2, color=c) 

    plt.show()

def lic(data, L, method):

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

    streamlines = calculate_streamlines(fun, method, positions, L)
    streamlines = np.reshape(streamlines, (X, Y, L*2 + 1, 2)).astype(int)
    
    for x in range(X):
        for y in range(Y):
            pixel_s = 0
            num_points = 0

            streamline = streamlines[x, y]

            for (s_x, s_y) in streamline:
                if s_x >= 0 and s_x < X and s_y >= 0 and s_y < Y:
                    pixel_s += blur[int(s_x), int(s_y)]
                    num_points += 1

            if num_points > 0:
                new_im[x][y] = pixel_s/num_points
            
            if np.linalg.norm(data[x][y]) < 1E-4:
                new_im[x][y] = 0
    
    plt.figure()
    plt.imshow(new_im)
    plt.show()

file1 = "data/isabel_2d.h5"
file2 = "data/metsim1_2d.h5"

data1 = read_data(file1)
data2 = read_data(file2)

#field_lines(data1, 10, 1000, "random", "RK4")
lic(data1, 20, "RK4")
