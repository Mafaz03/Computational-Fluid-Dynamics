import numpy as np
import matplotlib.pyplot as plt

def linear_to_flipped_coords(index, num_rows):
    col = index // num_rows
    row_from_bottom = index % num_rows
    row = num_rows - 1 - row_from_bottom
    return (row, col)

def find_neighbors(nodes, row, col, num_nodes_x, num_nodes_y):
    """!!!!!!!!!!!! Remeber that it returns node number not index !!!!!!!!!!!!"""
    north = nodes[row-1][col] if row > 0 else None
    south = nodes[row+1][col] if row < num_nodes_y - 1 else None
    west = nodes[row][col-1] if col > 0 else None
    east = nodes[row][col+1] if col < num_nodes_x - 1 else None
    return {'north': north, 'east': east, 'south': south, 'west': west}

def strech(axis, index, degrade):
    axis = np.array(axis, dtype=float)
    total = axis.sum()

    new_value = axis[index] * ((100 - degrade) / 100.0)

    leftover = total - new_value
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


def degrade_percentage(arr, index, degrade):
    assert index in (0, -1)

    total_sum = sum(arr)
    factor = (100 - degrade) / 100.0
    arr = np.array(arr, dtype=float)

    # Apply progressive decay
    if index == -1:
        for idx in range(1, len(arr)):
            arr[idx] = arr[idx-1] * factor
    else:
        for idx in range(len(arr)-2, -1, -1):
            arr[idx] = arr[idx+1] * factor

    # Scale to preserve total sum
    new_total_sum = arr.sum()
    arr *= total_sum / new_total_sum
    # print(arr)
    return arr


def plot_mesh(mesh, plot_size_y, plot_size_x):
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))

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
