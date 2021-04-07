
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import heapq


def read_data(filename):

    d1 = h5py.File(filename, 'r')

    xs = np.array(d1["Velocity"]["X-comp"])
    ys = np.array(d1["Velocity"]["Y-comp"])

    data = np.zeros((xs.shape[0], xs.shape[0], 2))

    data[:, :, 0] = xs
    data[:, :, 1] = ys

    return data
    

def normalize_data(data):
    norm = np.linalg.norm(data, axis=2)

    np.divide(data[:,:,0], norm, out=data[:,:,0], where=norm>1E-14)
    np.divide(data[:,:,1], norm, out=data[:,:,1], where=norm>1E-14)

def interpolate_2d(xs, ys, data):
    xcomp = RectBivariateSpline(xs, ys, data[:,:,0])
    ycomp = RectBivariateSpline(xs, ys, data[:,:,1])

    f = lambda x, y: np.array((xcomp(x, y, grid=False), ycomp(x, y, grid=False))).transpose()
    return f

def calculate_field_line_conditional(f, method, midpoint, length, filled, X, Y, N, h):
    L = 2*int(length/(2*h)) + 1

    cur_pos = midpoint
    field_line = [cur_pos]

    for ch in [-1, 1]:
        for i in range(L):
            cur_x = cur_pos[0]
            cur_y = cur_pos[1]
            
            if method=="RK4":
                k1 = ch*f(cur_x, cur_y)
                k2 = k3 = ch*f(cur_x + h/2, cur_y + h/2)
                k4 = ch*f(cur_x + h, cur_y + h)
                
                next_pos = cur_pos + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
            else:
                next_pos = cur_pos + ch*h*f(cur_x, cur_y)
            
            if np.linalg.norm(next_pos - cur_pos) < 1E-10:   # no point to continue if there is zero movement
                break

            cur_pos = next_pos

            cur_x = cur_pos[0]
            cur_y = cur_pos[1]
            
            if cur_x >= X or cur_x < 0 or cur_y > Y or cur_y < 0:
                break

            if filled[int(cur_x*N/X)][int(cur_y*N/Y)]:
                break
            
            if ch==-1:
                field_line = [cur_pos] + field_line
            else:
                field_line = field_line + [cur_pos]
        
        cur_pos = midpoint

    return field_line


def calculate_field_lines_vectorized(f, method, midpoints, length, h):
    L = 2*int(length/(2*h)) + 1

    field_line = np.zeros((len(midpoints), 2*L + 1, 2))

    cur_pos = midpoints[:]

    field_line[:,L] = cur_pos
    index = L - 1

    for ch in [-1, 1]:
        for i in range(L):
            cur_x = cur_pos[:,0]
            cur_y = cur_pos[:,1]
            if method=="RK4":
                k1 = ch*f(cur_x, cur_y)
                k2 = ch*f(cur_x + k1[:,0]/2, cur_y + k1[:,1]/2)
                k3 = ch*f(cur_x + k2[:,0]/2, cur_y + k2[:,1]/2)
                k4 = ch*f(cur_x + k3[:,0], cur_y + k3[:,1])
                
                cur_pos = cur_pos + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
            else:
                cur_pos = cur_pos + ch*h*f(cur_x, cur_y)
            
            field_line[:,index] = cur_pos
            index += ch
        
        cur_pos = midpoints[:]
        index = L + 1

    return field_line

def corner(X, Y, num_points, fun, method, L, h):

    points = np.array([[X, Y]])

    return points, calculate_field_lines_vectorized(fun, method, points, L, h)


def uniform(X, Y, num_points, fun, method, L, h):
    N_x = int(np.sqrt(num_points*X/Y))
    N_y = int(num_points/N_x)

    xs = np.linspace(0, X, N_x)
    ys = np.linspace(0, Y, N_y)

    ys, xs = np.meshgrid(xs, ys)
    xs = xs.flatten()
    ys = ys.flatten()

    num_points = N_x*N_y

    points = np.zeros((num_points, 2))
    points[:,0] = xs
    points[:,1] = ys
    
    return points, calculate_field_lines_vectorized(fun, method, points, L, h)


def random(X, Y, num_points, fun, method, L, h):
    xs = X*np.random.rand(num_points)
    ys = Y*np.random.rand(num_points)
    
    points = np.zeros((num_points, 2))
    points[:,0] = xs
    points[:,1] = ys

    return points, calculate_field_lines_vectorized(fun, method, points, L, h)

def density_based(X, Y, num_points, fun, method, L, h):
    density = X//100
    print("density: ", density)

    N = int(max(X, Y)/density)

    print("N: ", N)

    xs = np.linspace(0, X-1, N)
    ys = np.linspace(0, Y-1, N)

    filled = np.zeros((N, N), dtype=bool)
    queue = []

    for x in xs:
        for y in ys:
            heapq.heappush(queue, (0, x, y, []))
            #plt.plot(y, x, "o", alpha=0.2)
    
    next_queue = []
    field_lines = []
    threshold = 5

    points = []

    # now let's get started

    while len(queue)>0 and len(field_lines) < num_points:
        new_point = heapq.heappop(queue)[1:3]
        field_line = calculate_field_line_conditional(fun, method, new_point, L, filled, X, Y, N, h)
        field_lines.append(np.array(field_line))
        points.append(new_point)

        for (s_x, s_y) in field_line:              # covered
            filled[int(s_x/X*N)][int(s_y/Y*N)] = True

        for item in queue:
            weight = item[0]
            point_x = item[1]
            point_y = item[2]
            distances = item[3]
             
            if not filled[int(point_x*N/X)][int(point_y*N/Y)]:
                min_dist = min(np.linalg.norm(field_line - np.array([point_x, point_y]), axis=1))
                distances.append(min_dist)
                heapq.heappush(next_queue, (-min(distances), point_x, point_y, distances))

        print("filled: ", np.sum(filled))
        print("queue: ", len(next_queue))
        print("field_lines: ", len(field_lines))

        queue = next_queue
        next_queue = []

    return np.array(points), field_lines


def feature_based(X, Y, num_points, fun, method, L, h):

    # calculate a number of random points; sort according to vorticity, use the
    # num_points with highest values to display the vector field
    xs = X*np.random.rand(int(num_points*2))
    ys = Y*np.random.rand(int(num_points*2))

    ranked = []

    for (x, y) in zip(xs, ys):
        c = 0 #curl(fun, x, y)

        ranked.append((c, x, y))

    ranked.sort()
    ranked = np.array(ranked)

    points = np.zeros((num_points, 2))
    points[:,0] = ranked[:num_points,1]
    points[:,1] = ranked[:num_points,2]

    return points, calculate_field_lines_vectorized(fun, method, points, L, h)    


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


def lic(axis, data, length, method, blur, h):

    L = int(length/h)
    print("L: ", L)
    X, Y, _ = data.shape

    new_im = np.zeros_like(blur)
    
    xs = np.linspace(0, X-1, X)
    ys = np.linspace(0, Y-1, Y)
     
    fun = interpolate_2d(xs, ys, data)

    Xs, Ys = np.meshgrid(xs, ys)

    xs2 = np.transpose(Xs).flatten()
    ys2 = np.transpose(Ys).flatten()

    points = np.zeros((len(xs2), 2))
    points[:,0] = xs2
    points[:,1] = ys2

    # calculate field_lines

    X_b, Y_b = blur.shape

    field_lines = calculate_field_lines_vectorized(fun, method, points, L, h)

    _, L, _ = field_lines.shape

    field_lines[:,:,0] *= 1
    field_lines[:,:,1] *= 1

    field_lines = np.reshape(field_lines, (X, Y, L, 2))
    
    field_lines[:, :, :, 0] *= (X_b/X)
    field_lines[:, :, :, 1] *= (Y_b/Y)

    field_lines = field_lines.astype(int)

    wr = within_range(field_lines, X_b, Y_b)     # filter for out of bounds errors

    field_lines[:, :, :, 0] = np.where(wr, field_lines[:, :, :, 0], 0)
    field_lines[:, :, :, 1] = np.where(wr, field_lines[:, :, :, 1], 0)

    new_im = np.average(blur[field_lines[:, :, :, 0], field_lines[:, :, :, 1]], axis=2, weights=wr)
    

    # just a visual thing
    norm = np.linalg.norm(data, axis=2)
    new_im = np.where(norm > 1E-10, new_im, 0)   # make initial zero values zero

    axis.imshow(new_im)


def plot_seeding_vs_length(data1, data2):
    labels = ["Uniform", "Random", "Density based", "Feature based"]

    for (data, data_label) in zip([data1, data2], ["isabel", "metsim"]):
        fig1, axes1 = plt.subplots(4, 4, figsize=(10, 10))
        fig2, axes2 = plt.subplots(4, 4, figsize=(10, 10))
        
        X, Y, _ = data.shape

        for (i, seeding_strategy) in enumerate([density_based]): #uniform, random, density_based, feature_based]):
            N = [2*10, 2*40, 2*160, 2*640]

            for j in range(4):
                length = N[3-j]
                num_points = N[j]
                print(length, num_points)
                
                points, field_lines = calculate_field_lines(data, length, num_points, seeding_strategy, "RK4")
                
                for axes in [axes1, axes2]:
                    axes[i][j].set_xticks([])
                    axes[i][j].set_yticks([])
                    axes[-1][j].set_xlabel(f"N: {length}")
                    axes[i][0].set_ylabel(labels[i])
                
                plot_field_lines(axes1[i][j], data, field_lines)
                plot_seeding_points(axes2[i][j], data, points)

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


def plot_FE_vs_RK4_lic(data):
    length = 40
    X, Y, _ = data.shape
    blur = np.random.uniform(size=(X, Y))

    hvalues = [5, 1, 0.5]

    fig, axes = plt.subplots(len(hvalues), 2, figsize=(6, 5))

    for (i, h) in enumerate(hvalues):
        for (j, method) in enumerate(["FE", "RK4"]):
            axis = axes[i][j]
            lic(axis, data, length, method, blur, h)
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

    fig, axes = plt.subplots(2, len(lengths), figsize=(6, 5))
    
    for i in range(2):
        blur = np.random.uniform(size=((i+1)*X, (i+1)*Y))

        for (j, l) in enumerate(lengths):
            axis = axes[i][j]
            lic(axis, data, l, "RK4", blur, 1)

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

#plot_different_lengths_lic(data1)
#plot_FE_vs_RK4_lic(data1)
