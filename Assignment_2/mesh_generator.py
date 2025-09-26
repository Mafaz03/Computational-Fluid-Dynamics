import numpy as np
import matplotlib.pyplot as plt
from helper import *

## Inputs ###################

dx = 5
dy = 5

Lx = 100
Ly = 100

# For equidistant change it to 0, 0
x_percentage = 5 # reduce by {x_percentage}%
y_percentage = 5 # reduce by {y_percentage}%

index_x = -1
index_y = 0

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
        #    1
        # 0|   | 2
        #    3

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

        if y == 0: cell.edge_node_pos.append(3)
        if x == 0: cell.edge_node_pos.append(0)
        if x == (Ly/dy) - 1 : cell.edge_node_pos.append(2)
        if y == (Lx/dx) - 1 : cell.edge_node_pos.append(1)

        x_val += cell.Cell_size_x
    x_val = 0
    y_val += cell.Cell_size_y

def strech_mesh(mesh: np.array, x_percentage, y_percentage, strech_function, index_x, index_y):
    if x_percentage == 0 and y_percentage == 0: # Edge case
        return mesh
    
    # Stretch in x
    if x_percentage != 0:
        for y in range(mesh.shape[0]):
            x_array_size = [mesh[y][x].Cell_size_x for x in range(mesh.shape[1])]
            stretched_array_size = strech_function(x_array_size, index_x, x_percentage)
            for x in range(mesh.shape[1]):
                mesh[y][x].Cell_size_x = stretched_array_size[x]

    # Stretch in y
    if y_percentage != 0:
        for x in range(mesh.shape[1]):
            y_array_size = [mesh[y][x].Cell_size_y for y in range(mesh.shape[0])]
            stretched_array_size = strech_function(y_array_size, index_y, y_percentage)
            for y in range(mesh.shape[0]):
                mesh[y][x].Cell_size_y = stretched_array_size[y]

    # Recompute coordinates AFTER both
    y_val = 0
    for y in range(mesh.shape[0]):
        x_val = 0
        for x in range(mesh.shape[1]):
            mesh[y][x].Gx = x_val
            mesh[y][x].Gy = y_val
            x_val += mesh[y][x].Cell_size_x
        y_val += mesh[y][0].Cell_size_y

    return mesh 


mesh = strech_mesh(mesh, x_percentage, y_percentage, strech_function = degrade_percentage, index_x=index_x, index_y=index_y)   

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

        edge_node_pos = cell.edge_node_pos
        if len(edge_node_pos) != 0:
            if 0 in edge_node_pos:
                ax.add_patch(plt.Circle(
                    (cell.Gx, cell.Gy + cell.Cell_size_y / 2),
                    0.3, color='red', fill=False)
                )

            if 1 in edge_node_pos:
                ax.add_patch(plt.Circle(
                    (cell.Gx + cell.Cell_size_x / 2, cell.Gy + cell.Cell_size_y),
                    0.3, color='red', fill=False)
                )

            if 2 in edge_node_pos:
                ax.add_patch(plt.Circle(
                    (cell.Gx + cell.Cell_size_x, cell.Gy + cell.Cell_size_y / 2),
                    0.3, color='red', fill=False)
                )

            if 3 in edge_node_pos:
                ax.add_patch(plt.Circle(
                    (cell.Gx + cell.Cell_size_x / 2, cell.Gy),
                    0.3, color='red', fill=False)
                )
plt.show()