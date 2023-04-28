import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""" 
GENERIC FUNCTIONS
"""

# Rotation Matrices
def Rz(theta:float)->np.ndarray:
    """
    Rotation about Z Axis
    
    Adapted from Section 1.3.1 "Spacecraft Dynamics and Control", de Ruiter

    Args:
        theta (float): angle to rotate through [rad]

    Returns:
        np.ndarray: Rotation matrix about Z axis
    """
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def Ry(theta:float)->np.ndarray:
    """
    Rotation about Y Axis
    
    Adapted from Section 1.3.1 "Spacecraft Dynamics and Control", de Ruiter

    Args:
        theta (float): angle to rotate through [rad]

    Returns:
        np.ndarray: Rotation matrix about Y axis
    """
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rx(theta:float)->np.ndarray:
    """
    Rotation about X Axis
    
    Adapted from Section 1.3.1 "Spacecraft Dynamics and Control", de Ruiter

    Args:
        theta (float): angle to rotate through [rad]

    Returns:
        np.ndarray: Rotation matrix about X axis
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

