#!/usr/bin/env python3

import numpy as np

from numpy.typing import ArrayLike
from typing import Type

class quaternion:
    """A class that implements a representation of quaternions, as well as operations on them.

    Attributes
    ----------
    v : float
        The scalar part of the quaternion.
    u : array-like of shape (3,)
        The complex part of the quaternion.
    """
    def __init__(self: 'quaternion', v: float, u: ArrayLike) -> None:
        self.v = v
        self.u = u
        self.normalize()

    @classmethod
    def from_rotation_vector(cls: Type['quaternion'], rotation_vector: ArrayLike) -> 'quaternion':
        """Creates a quaternion from a rotation vector.

        Parameters
        ----------
        rotation_vector : array-like of shape (3,)
            The rotation vector, i.e. an axis-angle representation with the angle premultipled to the
            axis.

        Returns
        -------
        quaternion
            The quaternion representation of the rotation vector.
        """
        angle = np.linalg.norm(rotation_vector)
        axis = rotation_vector/angle
        v = np.abs(np.cos(angle/2))
        u = axis*np.sin(angle/2)
        return cls(v,u)

    @classmethod
    def from_array(cls: Type['quaternion'], array: ArrayLike) -> 'quaternion':
        """Creates a quaternion from an array.

        Parameters
        ----------
        array : array-like of shape (4,)
            Array holding the quaternion components. The scalar part should be the first element.

        Returns
        -------
        quaternion
            The corresponding quaternion
        """
        return cls(array[0],array[1:])
    
    @classmethod
    def exp(cls: Type['quaternion'], w: ArrayLike) -> 'quaternion':
        """Project an R^3 vector to quaternion space.

        Parameters
        ----------
        w : array-like of shape (3,)
            The vector to project.

        Returns
        -------
        quaternion
            The result of the projection.
        """
        norm = np.linalg.norm(w)
        v = 1
        u = np.zeros(3)
        if not np.allclose(w,np.zeros_like(w)):
            v = np.cos(norm)
            u = np.sin(norm)*w/norm
        return cls(v,u)

    def log(self: 'quaternion') -> ArrayLike:
        """Projects the quaternion to Euclidean space.

        Returns
        -------
        array-like of shape (3,)
            The projection in R^3 of self.
        """
        norm = np.linalg.norm(self.as_array())
        u = np.sign(self.v)*self.u/norm
        v = np.sign(self.v)*self.v/norm
        if not np.allclose(u,np.zeros_like(u)):
            return np.arccos(v)*u/np.linalg.norm(u)
        else:
            return np.zeros_like(u)
        
    def abs(self: 'quaternion') -> ArrayLike:
        """Compute the absolute value of the quaternion's components.

        Returns
        -------
        array-like of shape (4,)
            The vector of absolute values of self's components.
        """
        return np.concatenate(([np.abs(self.v)],np.abs(self.u)))
    
    def as_array(self: 'quaternion') -> ArrayLike:
        """Returns the quaternion's components as an array, with the scalar component first.

        Returns
        -------
        array-like of shape (4,)
            The array representing the quaternion.
        """
        return np.concatenate(([self.v],self.u))
    
    def normalize(self: 'quaternion') -> 'quaternion':
        """Normalizes self.

        Returns
        -------
        quaternion
            The result of the normalization.
        """
        norm = np.linalg.norm(self.as_array())
        self.u = self.u/norm
        self.v = self.v/norm
        return self
    
    def derivative(self: 'quaternion', wh: ArrayLike, qa: 'quaternion', dt: float) -> ArrayLike:
        """Computes the Euclidean projection of the derivative of self.

        Parameters
        ----------
        wh : ArrayLike
            The angular velocity associated to self.
        qa : quaternion
            The auxiliary quaternion.
        dt : float
            The timestep for the derivative.

        Returns
        -------
        ArrayLike
            The derivative of the Euclidean projection of self.
        """
        xi1 = ((self.exp(wh*dt/2)*self)*~qa).log()
        xi2 = (self*~qa).log()
        return (xi1 - xi2)/dt
    
    def __add__(self: 'quaternion', q: 'quaternion') -> 'quaternion':
        """Overload the '+' operator to handle quaternion-quaternion sum.

        Parameters
        ----------
        q : quaternion
            The second quaternion

        Returns
        -------
        quaternion
            The result of the addition self+q.
        """
        return quaternion(self.v+q.v,self.u+q.u)

    def __sub__(self: 'quaternion', q: 'quaternion') -> 'quaternion':
        """Overload the '+' operator to handle quaternion-quaternion subtraction.

        Parameters
        ----------
        q : quaternion
            The second quaternion

        Returns
        -------
        quaternion
            The result of the subtraction self-q.
        """
        return quaternion(self.v-q.v,self.u-q.u)

    def __mul__(self: 'quaternion', q: 'quaternion') -> 'quaternion':
        """Overload the '*' operator to handle quaternion-quaternion multiplication or 
        quaternion-scalar multiplication.

        Parameters
        ----------
        q : quaternion or float
            The quantity to multiply the quaternion by.

        Returns
        -------
        quaternion
            The result of the quaternion multiplication self*q.
        """
        if isinstance(q,quaternion):
            u = self.v*q.u + q.v*self.u + np.cross(self.u,q.u)
            v = self.v*q.v - self.u.T@q.u
        else:
            u = self.u*q
            v = self.v*q
        return quaternion(v,u)

    def __truediv__(self: 'quaternion', q: float) -> 'quaternion':
        """Overload the '/' operator to handle quaternion-scalar division.

        Parameters
        ----------
        q : float
            The quantity to divide the quaternion by.

        Returns
        -------
        quaternion
            The result of the division self/q
        """
        return quaternion(self.v/q,self.u/q)
    
    def __invert__(self: 'quaternion') -> 'quaternion':
        """Overload the '~' operator to handle quaternion conjugation.

        Returns
        -------
        quaternion
            The conjugate quaternion of self.
        """
        return quaternion(self.v,-self.u)
    
    def __neg__(self: 'quaternion') -> 'quaternion':
        """Overload the '-' unary operator to handle flipping the sign of a quaternion.

        Returns
        -------
        quaternion
            Self with opposite sign.
        """
        return quaternion(-self.v,-self.u)
    
    def __pos__(self: 'quaternion') -> 'quaternion':
        """Overload the '+' unary operator.

        Returns
        -------
        quaternion
            A copy of self.
        """
        return quaternion(self.v,self.u)
    
    def __getitem__(self: 'quaternion', index: int) -> float:
        """Overload subscription.

        Parameters
        ----------
        index : int
            The index of the desired quaternion component. The scalar part is at index 0, the complex
            part at indexes 1:3.

        Returns
        -------
        float
            The requested quaternion component.
        """
        if index == 0:
            return self.v
        else:
            return self.u[index-1]

    def __setitem__(self: 'quaternion', index: int, value: float) -> None:
        """Overload subscription.

        Parameters
        ----------
        index : int
            The index of the desired quaternion component. The scalar part is at index 0, the complex
            part at indexes 1:3.
        value : float
            The value to assign to the component.
        """
        if index == 0:
            self.v = value
        else:
            self.u[index-1] = value