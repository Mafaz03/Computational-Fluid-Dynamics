import numpy as np
import matplotlib.pyplot as plt
from helper import *
from helper import linear_to_flipped_coords, find_neighbors
from gauss_sadel import Gauss_Sadel
from tqdm import tqdm
import gradio as gr

## Inputs ###################

# dx = 0.05
# dy = 0.025

# Lx = 1
# Ly = 0.5

# num_nodes_x = int(Lx / dx)
# num_nodes_y = int(Ly / dy)

# # For equidistant change it to 0, 0
# x_percentage = 0 # reduce by {x_percentage}%
# y_percentage = 0 # reduce by {y_percentage}%

# index_x = 0
# index_y = -1

# ### Boundary Conditions
# # T South : Temperature | 15
# # T East  : Flux        | 5000
# # T North : Temperature | 10
# # T West  : Insulated   | 

# q_E       = 5000  # uniform heat generation

# T_N       = 10 # North Side
# T_S       = 15


class node:
    def __init__(self):

        self.idx = None # Idx starts with 1 (i messed up)
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

        self.del_xe = None
        self.del_xn = None
        self.del_xw = None
        self.del_xs = None

def Diffusion_2D(dx, dy, Lx,  Ly, x_percentage, y_percentage, index_x, index_y, q_E, T_N, T_S):
    dx = float(dx)
    dy = float(dy)
    Lx = float(Lx)
    Ly = float(Ly)
    x_percentage = float(x_percentage)
    y_percentage = float(y_percentage)
    index_x = int(index_x)
    index_y = int(index_y)
    q_E = float(q_E)
    T_N = float(T_N)
    T_S = float(T_S)


    num_nodes_x = int(Lx / dx)
    num_nodes_y = int(Ly / dy)

    ##############################


    # Initializing empty mesh
    mesh = np.array([[node() for x in range(num_nodes_x)] for y in range(num_nodes_y)])

    # Equidistant mesh gets created here
    x_val = 0
    y_val = 0

    # the origin of the x and y axis is on the bottom left
    idx = 0
    for y in range(mesh.shape[0]):
        for x in range(mesh.shape[1]):
            cell = mesh[y][x]
            cell.idx = idx
            cell.Gx = x_val
            cell.Gy = y_val

            if y == 0: cell.edge_node_pos.append(3)
            if x == 0: cell.edge_node_pos.append(0)
            if x == (Ly/dy) - 1 : cell.edge_node_pos.append(2)
            if y == (Lx/dx) - 1 : cell.edge_node_pos.append(1)

            x_val += cell.Cell_size_x
            idx += 1
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

    for y in range(num_nodes_y):
        for x in range(num_nodes_x):

            cell = mesh[y][x]
            if y == 0:
                cell.del_xn = cell.Cell_size_y/2
                cell.del_xs = mesh[y][x].Cell_size_y/2 + mesh[y+1][x].Cell_size_y/2
                if x == 0:
                    cell.del_xw = cell.Cell_size_x/2
                    cell.del_xe = mesh[y][x].Cell_size_x/2 + mesh[y][x+1].Cell_size_x/2

                elif x == len(mesh[0])-1:
                    cell.del_xe = cell.Cell_size_x/2  
                    cell.del_xw = mesh[y][x].Cell_size_x/2 + mesh[y][x-1].Cell_size_x/2
                
                else: # non edge case 
                    cell.del_xw = mesh[y][x].Cell_size_x/2 + mesh[y][x-1].Cell_size_x/2   
                    cell.del_xe = mesh[y][x].Cell_size_x/2 + mesh[y][x+1].Cell_size_x/2

            if y == mesh.shape[0]-1:
                cell.del_xs = cell.Cell_size_y/2 
                cell.del_xn = mesh[y][x].Cell_size_y/2 + mesh[y-1][x].Cell_size_y/2

                if x == 0:
                    cell.del_xw = cell.Cell_size_x/2
                    cell.del_xe = mesh[y][x].Cell_size_x/2 + mesh[y][x+1].Cell_size_x/2
            
                elif x == len(mesh[0])-1:
                    cell.del_xe = cell.Cell_size_x/2  
                    cell.del_xw = mesh[y][x].Cell_size_x/2 + mesh[y][x-1].Cell_size_x/2
                
                else: # non edge case 
                    cell.del_xw = mesh[y][x].Cell_size_x/2 + mesh[y][x-1].Cell_size_x/2   
                    cell.del_xe = mesh[y][x].Cell_size_x/2 + mesh[y][x+1].Cell_size_x/2

            else:
                cell.del_xn = mesh[y][x].Cell_size_y/2 + mesh[y-1][x].Cell_size_y/2   
                cell.del_xs = mesh[y][x].Cell_size_y/2 + mesh[y+1][x].Cell_size_y/2   
                if x == 0:
                    cell.del_xw = cell.Cell_size_x/2
                    cell.del_xe = mesh[y][x].Cell_size_x/2 + mesh[y][x+1].Cell_size_x/2

                elif x == mesh.shape[1]-1:
                    cell.del_xe = cell.Cell_size_x/2
                    cell.del_xw = mesh[y][x].Cell_size_x/2 + mesh[y][x-1].Cell_size_x/2   
                    
                else: # non edge case 
                    cell.del_xw = mesh[y][x].Cell_size_x/2 + mesh[y][x-1].Cell_size_x/2   
                    cell.del_xe = mesh[y][x].Cell_size_x/2 + mesh[y][x+1].Cell_size_x/2    

    delta_x = [] # size of the cell in x direction
    delta_y = [] # size of the cell in y direction

    for y in range(len(mesh)):
        delta_x_temp = []
        delta_y_temp = []
        for x in range(len(mesh[0])):
            cell = mesh[y][x]
            delta_x_temp.append(cell.Cell_size_x)
            delta_y_temp.append(cell.Cell_size_y)
        delta_x.append(delta_x_temp)
        delta_y.append(delta_x_temp)

    nodes = []
    n = 1
    for i in range(int(num_nodes_x)):
        l = []
        for j in range(int(num_nodes_y)):
            l.append(n)
            n+=1
        nodes.append(l)

    # nodes = nodes[::-1]
    nodes = np.fliplr(np.array(nodes)).transpose()
    nodes = np.array(nodes)

    for i in range(int(num_nodes_y)):
        for j in range(int(num_nodes_x)):
            mesh[i][j].idx = nodes[i][j]

    coef_matrix = np.zeros((num_nodes_y*num_nodes_x, num_nodes_y*num_nodes_x))
    Su_vector = np.zeros_like(coef_matrix[:,0])

    T_old = np.ones(num_nodes_x * num_nodes_y)  # initial guess, can be nonzero if known
    tolerance = 80
    max_iterations = 1

    for iteration in range(max_iterations):
        print(iteration)
        for node_idx in range(num_nodes_x * num_nodes_y):
            # print(f"Processing node {node_idx}")
            cell_x, cell_y = linear_to_flipped_coords(node_idx, num_nodes_y)
            cell_x, cell_y = int(cell_x), int(cell_y)

            # For mesh[y][x] indexing
            cell = mesh[cell_x][cell_y]
            # print(f"Coordinates: ({cell_x}, {cell_y})")

            nei_dict = find_neighbors(nodes, cell_x, cell_y, num_nodes_x, num_nodes_y)
            # print(f"Neighbors: {nei_dict}")

            # Initialize coefficients
            a_N = a_S = a_E = a_W = 0
            Su = 0

            # -----------------------------
            k_cell = 16 * ((cell.Gy + (cell.Cell_size_y/2)) / Ly + 1)
            Sp = (1.5 * cell.Cell_size_x * cell.Cell_size_y) / T_old[node_idx]
            Su = 0
            # Su += Sp * T_old[node_idx]
            # -----------------------------

            # +++++++ SOUTH boundary: Dirichlet T=15 +++++++
            if nei_dict['south'] is None and nei_dict['east'] is not None and nei_dict['west'] is not None:
                a_E = (k_cell * cell.Cell_size_y) / cell.del_xe
                a_N = (k_cell * cell.Cell_size_x) / cell.del_xn    
                a_W = (k_cell * cell.Cell_size_y) / cell.del_xw
                a_S = (k_cell * cell.Cell_size_x) / cell.del_xs   # half-cell treatment

                a_P = a_E + a_N + a_S + a_W + Sp
                Su += a_S * T_S

                coef_matrix[node_idx, node_idx] = a_P
                coef_matrix[node_idx, nei_dict['east'] - 1] = a_E
                coef_matrix[node_idx, nei_dict['north'] - 1] = a_N
                coef_matrix[node_idx, nei_dict['west'] - 1] = a_W

            # +++++++ NORTH boundary: Dirichlet T=10 +++++++
            elif nei_dict['north'] is None and nei_dict['east'] is not None and nei_dict['west'] is not None:
                a_E = (k_cell * cell.Cell_size_y) / cell.del_xe
                a_S = (k_cell * cell.Cell_size_x) / cell.del_xs
                a_W = (k_cell * cell.Cell_size_y) / cell.del_xw
                a_N = (k_cell * cell.Cell_size_x) / cell.del_xn  # half-cell treatment

                a_P = a_E + a_N + a_S + a_W + Sp
                Su += a_N * T_N

                coef_matrix[node_idx, node_idx] = a_P
                coef_matrix[node_idx, nei_dict['east'] - 1] = a_E
                coef_matrix[node_idx, nei_dict['south'] - 1] = a_S
                coef_matrix[node_idx, nei_dict['west'] - 1] = a_W

            # +++++++ EAST boundary: Neumann flux q=5000 +++++++
            elif nei_dict['east'] is None:
                a_N = (k_cell * cell.Cell_size_x) / cell.del_xn if nei_dict['north'] is None else (k_cell * cell.Cell_size_x) / cell.del_xn
                a_S = (k_cell * cell.Cell_size_x) / cell.del_xs if nei_dict['south'] is None else (k_cell * cell.Cell_size_x) / cell.del_xs
                a_W = (k_cell * cell.Cell_size_y) / cell.del_xw if nei_dict['west'] is not None else 0

                # Add Dirichlet contribution for top-right / bottom-right corners
                if nei_dict['north'] is None:
                    Su += a_N * T_N
                if nei_dict['south'] is None:
                    Su += a_S * T_S

                # Neumann flux contribution
                Su += -q_E * cell.Cell_size_y

                a_P = a_N + a_S + a_W + Sp  # no east neighbor

                coef_matrix[node_idx, node_idx] = a_P
                if nei_dict['north'] is not None:
                    coef_matrix[node_idx, nei_dict['north'] - 1] = a_N
                if nei_dict['south'] is not None:
                    coef_matrix[node_idx, nei_dict['south'] - 1] = a_S
                if nei_dict['west'] is not None:
                    coef_matrix[node_idx, nei_dict['west'] - 1] = a_W

            # +++++++ WEST boundary: Insulated +++++++
            elif nei_dict['west'] is None:
                if nei_dict['north'] is None:
                    # top-left
                    a_N = (k_cell * cell.Cell_size_x) / cell.del_xn # half-cell treatment
                    a_S = (k_cell * cell.Cell_size_x) / cell.del_xs
                    a_E = (k_cell * cell.Cell_size_y) / cell.del_xe
                
                    Su += a_N * T_N

                elif nei_dict['south'] is None:
                    # bottom-left
                    a_N = (k_cell * cell.Cell_size_x) / cell.del_xn
                    a_S = (k_cell * cell.Cell_size_x) / cell.del_xs # half-cell treatment
                    a_E = (k_cell * cell.Cell_size_y) / cell.del_xe
                    Su += a_S * T_S

                else:
                    # middle of west boundary
                    a_N = (k_cell * cell.Cell_size_x) / cell.del_xn
                    a_S = (k_cell * cell.Cell_size_x) / cell.del_xs
                    a_E = (k_cell * cell.Cell_size_y) / cell.del_xe

                a_W = 0  # insulated
                a_P = a_E + a_N + a_S + Sp

                coef_matrix[node_idx, node_idx] = a_P
                if nei_dict['north'] is not None:
                    coef_matrix[node_idx, nei_dict['north'] - 1] = a_N
                if nei_dict['south'] is not None:
                    coef_matrix[node_idx, nei_dict['south'] - 1] = a_S
                coef_matrix[node_idx, nei_dict['east'] - 1] = a_E

            # +++++++ INTERIOR node +++++++
            else:
                if nei_dict['north'] is not None:
                    a_N = (k_cell * cell.Cell_size_x) / cell.del_xn
                    coef_matrix[node_idx, nei_dict['north'] - 1] = a_N
                if nei_dict['south'] is not None:
                    a_S = (k_cell * cell.Cell_size_x) / cell.del_xs
                    coef_matrix[node_idx, nei_dict['south'] - 1] = a_S
                if nei_dict['east'] is not None:
                    a_E = (k_cell * cell.Cell_size_y) / cell.del_xe
                    coef_matrix[node_idx, nei_dict['east'] - 1] = a_E
                if nei_dict['west'] is not None:
                    a_W = (k_cell * cell.Cell_size_y) / cell.del_xw
                    coef_matrix[node_idx, nei_dict['west'] - 1] = a_W

                a_P = a_E + a_N + a_S + a_W + Sp
                coef_matrix[node_idx, node_idx] = a_P

            # +++++++ Update source vector +++++++
            Su_vector[node_idx] = Su
            # print(a_P, end="\n\n")


        # Making non diagnol element negetive
        mask = ~np.eye(coef_matrix.shape[0], dtype=bool)
        coef_matrix[mask] = -coef_matrix[mask]

        X, loops_taken, relative_error_list = Gauss_Sadel(coef_matrix, 
                                                        Su_vector, 
                                                        np.zeros_like(Su_vector), # Initial Guess
                                                        tolerance=1e-4, max_loops=200)

        max_diff = np.max(np.abs(X - T_old))
        print(max_diff)
        if max_diff < tolerance:
            print(f"Converged in {iteration+1} outer iterations.")
            break
        T_old = X.copy()

    # Reshape for 5×5 grid
    X_grid = X.reshape(num_nodes_x, num_nodes_y).transpose()

    # Create coordinates  
    x = np.linspace(0, num_nodes_x*dx, num_nodes_x)
    y = np.linspace(0, num_nodes_y*dy, num_nodes_y)
    X_coords, Y_coords = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8,8))
    contour = ax.contourf(X_coords, Y_coords, X_grid, levels=20, cmap="inferno")
    fig.colorbar(contour, ax=ax, label="Temperature")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Temperature Distribution")
    ax.set_aspect("equal")
    ax.set_aspect(dx/dy)
                  
    return fig
    # fig, ax = plt.subplots(figsize=(8,8))
    # plt.figure(figsize=(8,8))
    # contour = plt.contourf(X_coords, Y_coords, X_grid, levels=20, cmap='inferno')
    # plt.colorbar(label='Temperature')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Temperature Distribution')

    # plt.gca().set_aspect(dx/dy) 
    # plt.show()

# Diffusion_2D(dx, dy, Lx,  Ly, x_percentage, y_percentage, index_x, index_y, q_E, T_N, T_S)

# radio Interface
demo = gr.Interface(
    fn=Diffusion_2D,
    inputs=[
        gr.Textbox(label="dx"),
        gr.Textbox(label="dy"),
        gr.Textbox(label="Lx"),
        gr.Textbox(label="Ly"),
        gr.Textbox(label="x_percentage"),
        gr.Textbox(label="y_percentage"),
        gr.Textbox(label="index_x"),
        gr.Textbox(label="index_y"),
        gr.Textbox(label="q_E"),
        gr.Textbox(label="T_N"),
        gr.Textbox(label="T_S"),
    ],
    outputs=gr.Plot(label="Temperature Contour"),
)

if __name__ == "__main__":
    demo.launch()