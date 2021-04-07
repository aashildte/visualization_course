
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


def define_visited_array(X, Y, density):
    N = int(max(X, Y)/density)

    print("N: ", N)

    xs = np.linspace(0, X-1, N)
    ys = np.linspace(0, Y-1, N)

    return xs, ys, np.zeros((N, N), dtype=bool)


def density_based(X, Y, num_points, fun, method, L, h):
    density = X//100
    xs, ys, filled = define_visited_array(X, Y, density)

    queue = []

    for x in xs:
        for y in ys:
            heapq.heappush(queue, (0, x, y, []))
    
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


def source_sink(x, y, N=20, dist=20):
    theta = np.linspace(0, 2*np.pi, N+1)
 
    points = np.zeros((N, 2))
    points[:,0] = x + dist*np.cos(theta[:N])
    points[:,1] = y + dist*np.sin(theta[:N])

    return points

def spiral(x, y, N = 20, dist=20):
    points = np.zeros((N, 2))

    points[:,0] = x + np.linspace(0, dist, N)
    points[:,1] = y

    return points

def saddle(x, y, N = 20, dist=20):
    points = np.zeros((N, 2))

    points[:,0] = x + np.linspace(-dist/2, dist/2, N)
    points[:,1] = y + np.linspace(-dist/2, dist/2, N)

    return points


def calculate_jacobian(f, x, y):
    h=1E-10
    dx = (f(x + h, y) - f(x - h, y))/2*h
    dy = (f(x, y + h) - f(x, y - h))/2*h

    return np.transpose(np.array([dx, dy]))

def feature_based(X, Y, num_points, fun, method, L, h):

    # find iteratively a point in the grid which is not visited
    # and has the lowest velocity (will start by filling up the
    # field with no movement at all)
    
    density = X//50
    xs, ys, filled = define_visited_array(X, Y, density)
    
    queue = []

    for x in xs:
        for y in ys:
            norm = np.linalg.norm(fun(x, y))
            if norm > 1E-5:
                heapq.heappush(queue, (norm, x, y))
            print("norm: ", norm)
    next_queue = []
    field_lines = []
    threshold = 5

    points = []


    while len(queue)>0 and len(field_lines) < num_points*20:
        new_point = heapq.heappop(queue)[1:3]
        x, y = new_point
        J = calculate_jacobian(fun, x, y)
        det = np.linalg.norm(J)

        if np.linalg.norm(fun(x, y)) > 1E-10:
            (l1, l2), _ = np.linalg.eig(J)
            
            if np.iscomplex(l1):
                print("complex!")
                template = source_sink(x, y)
                if l1.real > 0:
                    sign = 1
                else:
                    sign = -1
            else:
                if l1 > 0 and l2 > 0:
                    print("spiral")
                    template = spiral(x, y)
                elif l1 < 0 and l2 < 0:
                    print("spiral")
                    template = spiral(x, y)
                else:
                    print("saddle")
                    template = saddle(x, y)
            print("template: ", template)
            for point in template:
                field_line = calculate_field_line_conditional(fun, method, point, L, filled, X, Y, num_points, h)
                field_lines.append(np.array(field_line))
                points.append(new_point)

            for (s_x, s_y) in field_line:              # covered
                filled[int(s_x/X*num_points)][int(s_y/Y*num_points)] = True

            print("filled: ", np.sum(filled))
            print("queue: ", len(next_queue))
            print("field_lines: ", len(field_lines))
    
    points = np.array(points)
    field_lines = np.array(field_lines)

    return points, field_lines


def lic(data, length, method, blur, h):

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
    
    norm = np.linalg.norm(data, axis=2)
    new_im = np.where(norm > 1E-10, new_im, 0)   # make initial zero values zero

    return new_im
