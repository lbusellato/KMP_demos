#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import scipy.io as sio

from matplotlib.widgets import Button, TextBox

# Reads a MATLAB struct containing an array of cells containing a struct
def read_struct(filepath, cell_array_name='demos', fields=['pos','vel','acc'], max_cell=math.inf):
    # Load the MATLAB file containing the struct
    mat = sio.loadmat(filepath)
    # Extract the field from the struct
    cell_array = mat[cell_array_name][0]
    # Loop through the cells in the demos cell array
    struct_array = []
    for (i,cell) in enumerate(cell_array):
        if i >= max_cell:
            break
        # Extract the fields from the struct
        struct = {}
        for field in fields:
            struct[field] = cell[field][0][0]
        struct_array.append(struct)
    return struct_array