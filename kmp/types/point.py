#!/usr/bin/env python3

import numpy as np

from dataclasses import dataclass

@dataclass
class Point:
    """Class that defines the representation of each point in the demonstration
    database.

    Attributes
    ----------
    time : float
        The timestamp associated to the point.
    pose : array-like of shape (6,)
        The vector containing cartesian position and the rotation vector.
    twist : array-like of shape (6,)
        The vector containing linear and angular velocity.
    quat : array-like of shape (4,)
        The quaternion representation of the rotation vector.
    quat_eucl : array_like of shape (4,)
        The Euclidean projection of the quaternion.
    wrench : array-like of shape (6,)
        The vector containing forces and moments.
    """
    time: float = 0.0
    pose: np.ndarray = np.zeros(6)
    twist: np.ndarray = np.zeros(6)
    quat: np.ndarray = np.zeros(4)
    quat_eucl: np.ndarray = np.zeros(3)
    wrench: np.ndarray = np.zeros(6)