import numpy as np

def stretch_one_point_gradual(arr, index, percent):
    """
    Stretch or shrink one point by `percent`, and reduce others progressively.
    Total sum remains the same.
    """
    arr = np.array(arr, dtype=float)
    total = arr.sum()
    n = len(arr)

    # Compute new value for chosen index
    new_value = arr[index] * (1 + percent / 100.0)

    # Remaining total for other values
    leftover = total - new_value

    # Progressive weights for others
    weights = np.array([i+1 for i in range(n)], dtype=float)
    weights[index] = 0

    # Normalize weights so others sum to leftover
    factor = leftover / weights.sum()
    for i in range(n):
        if i != index:
            arr[i] = factor * weights[i]
    arr[index] = new_value

    return arr

# Testing
if __name__ == "__main__":

    x_axis = [5, 5, 5, 5, 5, 5]
    result = stretch_one_point_gradual(x_axis, index=0, percent=-90)  # Stretch 0th element by {percent}%
    print(result)
    print("Sum:", sum(result))