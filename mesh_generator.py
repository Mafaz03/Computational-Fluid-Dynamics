import numpy as np
import matplotlib.pyplot as plt
from helper import *

## Inputs ###################

dx = 5
dy = 5

Lx = 200
Ly = 200

degrade = 110 # reduce by 90%
index_x = 39
index_y = 39

##############################

class node:
    def __init__(self):
        
        self.node_pos = None
        self.Cell_size_x = 5
        self.Cell_size_y = 5
        self.grid_face = None
        self.node_type = None 

        self.Gx = None # Location along the x axis
        self.Gy = None # Location along the y axis

        self.edge_color = "blue"

        self.edge_node_pos = []

# Initializing empty mesh
mesh = np.array([[node() for x in range(int(Lx/dx))] for y in range(int(Ly/dy))])

# Equidistant mesh gets created here
x_val = 0
y_val = 0

# the origin of the x and y axis is on the bottom left
for y in range(mesh.shape[0]):
    for x in range(mesh.shape[1]):
        cell = mesh[y][x]
        cell.Gx = x_val
        cell.Gy = y_val

        x_val += cell.Cell_size_x
    x_val = 0
    y_val += cell.Cell_size_y

# def strech_mesh(mesh: np.array, percentage, strech_function, index_x, index_y):
#     for y in range(mesh.shape[0]):
#         # Extract current row cell sizes
#         x_array_size = [mesh[y][x].Cell_size_x for x in range(mesh.shape[1])]

#         # Stretch chosen index
#         stretched_array_size = strech_function(x_array_size, index_x, percentage)

#         # Assign back the new sizes and recompute bottom-left
#         gx_temp = 0
#         for x in range(mesh.shape[1]):
#             mesh[y][x].Cell_size_x = stretched_array_size[x]
#             mesh[y][x].Gx = gx_temp   # bottom-left of this cell
#             gx_temp += stretched_array_size[x]

#     for x in range(mesh.shape[1]):
#         # Extract current column cell sizes
#         y_array_size = [mesh[y][x].Cell_size_y for y in range(mesh.shape[0])]

#         # Stretch chosen index 
#         stretched_array_size = strech_function(y_array_size, index_y, percentage)

#         # Assign back and recompute bottom-left Gy
#         gy_temp = 0
#         for y in range(mesh.shape[0]):
#             mesh[y][x].Cell_size_y = stretched_array_size[y]
#             mesh[y][x].Gy = gy_temp
#             gy_temp += stretched_array_size[y]

#     return mesh  

def strech(axis, index, degrade):
    axis = np.array(axis, dtype=float)
    total = axis.sum()

    # Step 1: shrink chosen index
    new_value = axis[index] * ((100 - degrade) / 100.0)

    # Step 2: leftover
    leftover = total - new_value

    # Step 3: distribute leftover based on distance from index
    n = len(axis)
    distances = np.array([abs(i - index) for i in range(n)], dtype=float)
    weights = (distances)    # farther gets more
    weights[index] = 0

    factor = leftover / weights.sum()

    new_axis = np.zeros_like(axis)
    for i in range(n):
        if i == index:
            new_axis[i] = new_value
        else:
            new_axis[i] = factor * weights[i]

    return new_axis.tolist()

mesh = strech_mesh(mesh, degrade, strech_function = strech, index_x=index_x, index_y=index_y)   

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

plot_size_y = Ly
plot_size_x = Lx

canvas = np.ones((int(plot_size_y), int(plot_size_x)))
ax.imshow(canvas, origin='lower', cmap='gray',
          extent=[0, plot_size_x, 0, plot_size_y])

for y in range(mesh.shape[0]):
    for x in range(mesh.shape[1]):
        cell = mesh[y][x]

        circle = plt.Circle(
                (cell.Gx + cell.Cell_size_x / 2, cell.Gy + cell.Cell_size_y / 2),
                0.1, color='red', fill=False
            )
        ax.add_patch(circle)

        rect = plt.Rectangle((cell.Gx, cell.Gy), cell.Cell_size_x, cell.Cell_size_y, linewidth=1, edgecolor=cell.edge_color, facecolor='none')
        ax.add_patch(rect)
plt.show()