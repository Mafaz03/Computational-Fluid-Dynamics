import numpy as np


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
