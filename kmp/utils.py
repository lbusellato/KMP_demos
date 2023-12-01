#!/usr/bin/env python3

import csv
import math
import numpy as np
import os
import scipy.io as sio

from kmp.types.point import Point
from kmp.types.quaternion import quaternion
    
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
        
# Finds the index of the element in list closest to val
def find_closest_index(list, val):
    smallest_dist = np.inf
    for i,l in enumerate(list.T):
        dist = np.linalg.norm(val-l)
        if dist < smallest_dist:
            closest_index = i
            smallest_dist = dist
    return closest_index

# Extracts the data containing the pose from the UR5 dataset in a format usable by the code
def create_dataset(path,subsample=100):
    cols = []
    out = []
    prev_quat = None
    qa = None
    dt = 0.01/subsample
    sign = 1
    files = [f for f in os.listdir(path) if f != 'pose_data.npy' and os.path.isfile(os.path.join(path, f))]
    for file in files:
        with open(os.path.join(path, file)) as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for (j,row) in enumerate(reader):
                if j%subsample == 0 and j <= 7500:
                    if len(cols)==0:
                        # Get the columns of the relevant values in the csv
                        column_names = ['actual_TCP_pose_','actual_TCP_speed_','actual_TCP_force_']
                        cols = [i for i, val in enumerate(row) if any([s in val for s in column_names])]
                    elif j != 0:
                        row = np.array(row,dtype=float)
                        t = j*dt
                        pose = row[cols[:6]]
                        twist = row[cols[6:12]]
                        # Convert from axis-angle to quaternion
                        quat = quaternion.from_rotation_vector(row[cols[3:6]])
                        # Recover the auxiliary quaternion
                        if qa is None:
                            qa = quat
                        # Handle representation ambiguities
                        if prev_quat is not None:
                            max_prev = np.argmax(quat.abs())
                            max_curr = np.argmax(prev_quat.abs())
                            # Flip the quaternion sign if its biggest component has done so
                            if max_prev == max_curr and np.sign(quat[max_curr]) != np.sign(prev_quat[max_prev]):
                                sign = -sign
                        # Project to euclidean space
                        quat_eucl = (quat*~qa).log()
                        wrench = row[cols[-6:]]
                        out.append(Point(t,pose,twist,-quat*sign,quat_eucl,wrench))
                        prev_quat = quat
    np.save(os.path.join(path + "pose_data.npy"), out)
    return out