#!/usr/bin/env python3

import csv
import math
import numpy as np
import os
import scipy.io as sio

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

# Computes the conjugate of a quaternion
def quat_conj(q):
    q[-3:] = -q[-3:]
    return q

# Computes the product of two quaternions
def quat_mul(q1,q2):
    u = q1[0]*q2[1:] + q2[0]*q1[1:] + np.cross(q1[1:],q2[1:])
    v = q1[0]*q2[0]-q1[1:]@q2[1:].T
    return np.array([v,u[0],u[1],u[2]])

# Project a quaternion in Euclidean space
def log(q):
    norm = np.linalg.norm(q)
    u = q[1:]/norm
    v = q[0]/norm
    if v < 0:
        u = -u
        v = -v
    if not np.allclose(u,np.zeros_like(u)):
        return np.arccos(v)*u/np.linalg.norm(u)
    else:
        return np.zeros_like(u)

# Projects a Euclidean vector in quaternion space
def exp(w):
    norm = np.linalg.norm(w)
    if not np.allclose(w,np.zeros_like(w)):
        q = np.array([np.cos(norm),np.sin(norm)*w[0]/norm,np.sin(norm)*w[1]/norm,np.sin(norm)*w[2]/norm])
    else:
        q = np.array([1,0,0,0])
    return q/np.linalg.norm(q)

# Extracts the data containing the pose from the UR5 dataset in a format usable by the code
def create_dataset(path,subsample=100):
    cols = None
    out = None
    prev_quat = None
    qa = None
    dt = 0.01/subsample
    sign = 1
    for file in os.listdir(path):
        if file != 'pose_data.npy' and os.path.isfile(os.path.join(path, file)):
            with open(os.path.join(path, file)) as csv_file:
                reader = csv.reader(csv_file, delimiter=' ')
                for (j,row) in enumerate(reader):
                    if j%subsample == 0 and j <= 8000:
                        if cols is None:
                            # Get the columns of the relevant values in the csv
                            cols = [i for i, val in enumerate(row) if "actual_TCP_pose_" in val]
                        elif j != 0:
                            data = np.array([row[i] for i in cols]).astype(float)
                            # Auxiliary quaternion (conjugate)
                            if qa is None:
                                qa = quat_conj(data[-4:])
                            # Recover the angle
                            angle = np.linalg.norm(np.array(data[-3:]))
                            axis = np.array(data[-3:])/angle
                            # Convert from axis-angle to quaternion
                            qs = np.abs(np.cos(angle/2))
                            qx = axis[0]*np.sin(angle/2)
                            qy = axis[1]*np.sin(angle/2)
                            qz = axis[2]*np.sin(angle/2)
                            quat = np.array([qs,qx,qy,qz])
                            quat = quat/np.linalg.norm(quat)
                            # Handle representation ambiguities
                            if prev_quat is not None:
                                quat_abs = np.abs(quat)
                                prev_quat_abs = np.abs(prev_quat)
                                max_prev = np.argmax(quat_abs)
                                max_curr = np.argmax(prev_quat_abs)
                                # Flip the quaternion sign if its biggest component has done so
                                if max_prev == max_curr:
                                    if np.sign(quat[max_curr]) != np.sign(prev_quat[max_prev]):
                                        sign = -sign
                                    else:
                                        sign = sign
                            # Project to euclidean space
                            quat_prod = quat_mul(quat,qa)
                            quat_eucl = log(quat_prod)
                            if out is None:
                                out = np.array([j*dt,data[0],data[1],data[2],
                                                quat_eucl[0],quat_eucl[1],quat_eucl[2],
                                                quat[0],quat[1],quat[2],quat[3]])
                            else:
                                out = np.vstack((out,np.array([j*dt,data[0],data[1],data[2],
                                                               quat_eucl[0],quat_eucl[1],quat_eucl[2],
                                                                quat[0]*sign,quat[1]*sign,quat[2]*sign,quat[3]*sign])))
                            prev_quat = quat
    np.save(os.path.join(path + "pose_data.npy"), out)
    return out
