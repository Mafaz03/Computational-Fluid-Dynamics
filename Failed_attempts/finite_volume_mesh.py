import numpy as np
import matplotlib.pyplot as plt

dx = 5
dy = 5

Lx = 500
Ly = 500

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

mesh = []
for y in range(int(Ly/dy)):
    mesh.append([node() for x in range(int(Lx/dx))])
# Basically the shape is (Lx / dx) x (Ly x dx)
mesh = np.array(mesh)

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


plot_size_x = 0
plot_size_y = 0

run_x = True

for y in range(mesh.shape[0]):
    if run_x == True:
        for x in range(mesh.shape[1]):
            plot_size_x += mesh[y][x].Cell_size_x
    run_x = False

    plot_size_y += mesh[y][x].Cell_size_y


fig, ax = plt.subplots(figsize=(10, 10))  # width=10 inches, height=10 inches

canvas = np.ones((plot_size_y, plot_size_x))
ax.imshow(canvas, origin='lower', cmap='gray',
          extent=[0, plot_size_x, 0, plot_size_x])

for y in range(mesh.shape[0]):
    for x in range(mesh.shape[1]):
        cell = mesh[y][x]

        # Draw circle at cell center
        circle = plt.Circle((cell.Gx + cell.Cell_size_x/2, cell.Gy + cell.Cell_size_y/2), 0.1, color='red', fill=False)
        ax.add_patch(circle)

        # Draw rectangle for each cell
        rect = plt.Rectangle((cell.Gx, cell.Gy), dx, dy, linewidth=1, edgecolor=cell.edge_color, facecolor='none')
        ax.add_patch(rect)

plt.show()