'''
In this file, we perform the fourier feature map described in Ref 3, which is used as input for LASSO regression (in functions.py file)
'''

import numpy as np


def get_local_edges(input_edge, dim_grid, delta=1):
    '''
    Given an input edge where the observable (2 qubit correlation) acts on, the dimensions of the qubit lattice, and the maximum
    Manhattan distance, we return two vectors indicating which horizontal and vertical couplings are inside the local area

    Inputs
    -------
    input_edge: tuple of two tuples
        Tuple containing the coordinates of two qubits

    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

    delta: int
        Maximum Manhattan distance for selecting edges closed to the input_edge

    Outputs
    --------
    local_horizontal: np.ndarray
        Array of length len(J_right.flatten()) with 1s where that coupling must be selected, and 0s when it must not
    
    local_vertical: np.ndarray
        Array of length len(J_down.flatten()) with 1s where that coupling must be selected, and 0s when it must not
    
    '''

    rows, cols = dim_grid

    def edge_midpoint(edge):
        (x1, y1), (x2, y2) = edge
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def l1_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # Initialize edge matrices
    horizontal_edges = np.empty((rows, cols - 1), dtype=object)
    vertical_edges = np.empty((rows - 1, cols), dtype=object)

    # Fill edge matrices and collect midpoints
    for x in range(rows):
        for y in range(cols):
            if y + 1 < cols:
                horizontal_edges[x, y] = ((x, y), (x, y + 1))
            if x + 1 < rows:
                vertical_edges[x, y] = ((x, y), (x + 1, y))

    # Get midpoint of the input edge
    center = edge_midpoint(input_edge)

    # Compute local masks
    local_horizontal = np.zeros((rows, cols - 1), dtype=bool)
    for x in range(rows):
        for y in range(cols - 1):
            if horizontal_edges[x, y] is not None:
                mid = edge_midpoint(horizontal_edges[x, y])
                local_horizontal[x, y] = l1_distance(mid, center) <= delta

    local_vertical = np.zeros((rows - 1, cols), dtype=bool)
    for x in range(rows - 1):
        for y in range(cols):
            if vertical_edges[x, y] is not None:
                mid = edge_midpoint(vertical_edges[x, y])
                local_vertical[x, y] = l1_distance(mid, center) <= delta

    return local_horizontal, local_vertical



def generate_local_couplings(J_right, J_down, local_h, local_v):
    '''
    Generate the flat vector of couplings close ot the edge where the correlation observable acts on

    Inputs
    -------
    J_right: np.ndarray
        Horizontal couplings
        Dimension dim_grid[0] x (dim_grid[1]-1)

    J_down: np.ndarray
        Vertical couplings
        Dimension (dim_grid[0]-1) x dim_grid[1]
    
    local_h: np.ndarray
        Array of length len(J_right.flatten()) with 1s where that coupling must be selected, and 0s when it must not

    local_v:np.ndarray
        Array of length len(J_down.flatten()) with 1s where that coupling must be selected, and 0s when it must not

    Output
    -------
    z: np.ndarray
        Flat vector with the selected local couplings
    '''

    J_right_local = J_right[local_h]
    J_down_local = J_down[local_v]
    z = np.concatenate((J_right_local, J_down_local))

    return z



def random_fourier_feature_map(Z, w, gamma, R):
    '''
    Perform the random Fourier feature map as described in the Reference 3 (Imporved machine learning algorithm for predicting ground state properties)

    Inputs
    -------
    Z: np.ndarray
        Flat vector with the selected local couplings
    
    w: np.ndarray
        Random vector of standard distribution related to the frequency of the Fourier map
    
    gamma: float
        Hyperparameter of the Fourier map that is related to its frequency
    
    R: int
        Hyperparameter of Fourier map related to ita dimension/complexity

    Outputs
    --------
    phi: np.ndarray
        Array of length 2*R*number of local couplings
        
    '''

    # Perform random Fourier feature map as described in the Reference 3 (Imporved machine learning algorithm for predicting ground state properties)
    l = len(Z)
    p = gamma/np.sqrt(l)
    phi = np.zeros(l*2*R)
    for i, z in enumerate(Z):
        for j in range(2*R):
            if j % 2 == 0:
                phi[i*2*R+j] = np.cos(p*w[j//2]*z)
            else:
                phi[i*2*R+j] = np.sin(p*w[j//2]*z)

    return phi


    




