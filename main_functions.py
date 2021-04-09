
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
                k2 = ch*f(cur_x + k1[0]/2, cur_y + k1[1]/2)
                k3 = ch*f(cur_x + k2[0]/2, cur_y + k2[1]/2)
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

            if filled[int(cur_x/X*N)][int(cur_y/Y*N)]:
                break
            
            if ch==-1:
                field_line = [cur_pos] + field_line
            else:
                field_line = field_line + [cur_pos]
        
        cur_pos = midpoint

    return np.array(field_line)


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
    density = 0.5
    N = int(X/density)
    _, _, filled = define_visited_array(X, Y, density)
    
    N_x = int(np.sqrt(num_points*X/Y))
    N_y = int(num_points/N_x)

    xs = np.linspace(0, X, N_x + 2)[1:-1]
    ys = np.linspace(0, Y, N_y + 2)[1:-1]

    ys, xs = np.meshgrid(xs, ys)
    xs = xs.flatten()
    ys = ys.flatten()

    num_points = N_x*N_y

    points = np.zeros((num_points, 2))
    points[:,0] = xs
    points[:,1] = ys
   
    field_lines = []

    for new_point in points:
        x, y = new_point
        if not filled[int(x/X*N)][int(y/Y*N)]:
            field_line = calculate_field_line_conditional(fun, method, new_point, L, filled, X, Y, N, h)
            field_lines.append(field_line)
            
            for (s_x, s_y) in field_line:              # covered
                filled[int(s_x/X*N)][int(s_y/Y*N)] = True

    return points, field_lines


def random(X, Y, num_points, fun, method, L, h):
    density = 0.5
    N = int(X/density)
    _, _, filled = define_visited_array(X, Y, density)

    field_lines = []
    points = []

    while len(field_lines) < num_points and np.sum(filled) < N*N:
        x = X*np.random.rand()
        y = Y*np.random.rand()
        new_point = np.array([x, y])

        if not filled[int(x/X*N)][int(y/Y*N)]:
            field_line = calculate_field_line_conditional(fun, method, new_point, L, filled, X, Y, N, h)
            field_lines.append(field_line)
            points.append(new_point)
            for (s_x, s_y) in field_line:              # covered
                filled[int(s_x/X*N)][int(s_y/Y*N)] = True

            print(np.sum(filled), len(field_lines))
    
    return np.array(points), field_lines


def define_visited_array(X, Y, density):
    N = int(max(X, Y)/density)

    print("N: ", N)

    xs = np.linspace(0, X-1, N)
    ys = np.linspace(0, Y-1, N)

    return xs, ys, np.zeros((N, N), dtype=bool)


def density_based(X, Y, num_points, fun, method, L, h):
    density = 0.5
    N = int(X/density)
    xs, ys, filled = define_visited_array(X, Y, density)

    queue = []

    for x in np.linspace(1, X-1, num_points):
        for y in np.linspace(1, Y-1, num_points):
            if np.linalg.norm(fun(x, y)) > 1E-10:
                heapq.heappush(queue, (0, x, y, []))

    next_queue = []
    field_lines = []
    threshold = 5

    points = []

    # now let's get started

    while len(queue)>0 and len(field_lines) < num_points:
        new_point = heapq.heappop(queue)[1:3]
        field_line = calculate_field_line_conditional(fun, method, new_point, L, filled, X, Y, N, h)
        
        if len(field_line) > 0 and abs(field_line[0][0] - field_line[-1][0]) > 1 and \
                abs(field_line[0][1] - field_line[-1][1]) > 1:

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

def within_range(field_lines, X, Y):
    s_x_min = field_lines[:, :, :, 0] >= 0
    s_x_max = field_lines[:, :, :, 0] < X
    
    s_y_min = field_lines[:, :, :, 1] >= 0
    s_y_max = field_lines[:, :, :, 1] < Y
    
    s_x = np.logical_and(s_x_min, s_x_max)
    s_y = np.logical_and(s_y_min, s_y_max)

    return np.logical_and(s_x, s_y)

def lic(data, length, method, blur, h):

    L = int(length/h)
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

    print(field_lines.shape)
    print(wr.shape)
    im_all = blur[field_lines[:, :, :, 0], field_lines[:, :, :, 1]]
    im_mid = blur[field_lines[:, :, 1:-1, 0], field_lines[:, :, 1:-1, 1]]
    print(blur.shape)
    print(field_lines[:, :, :, 0].shape)

    new_im = np.average(im_all, axis=2, weights=wr) + np.average(im_mid, axis=2, weights=wr[:,:,1:-1])

    #new_im = np.average(blur[field_lines[:, :, :, 0], field_lines[:, :, :, 1]], axis=2, weights=wr)
    
    norm = np.linalg.norm(data, axis=2)
    new_im = np.where(norm > 1E-10, new_im, 0)   # make initial zero values zero

    return new_im
