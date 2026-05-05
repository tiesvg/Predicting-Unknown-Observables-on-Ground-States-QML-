'''
In this file we define the parameters used for the simulation (both in main.py and functions.py)
'''

# General parameters
dim_grid = (2,2)
obs_name = 'correlation'
qubits = ((0,0), (0,1))
hamiltonian_label = 'heisenberg'

# Training parameters
num_examples = 100

# VQE parameters
depth = 3
opt_steps = 500
learning_rate = 0.01

# Random fourier map parameters
delta = 1
gamma = 0.6
R = 10

# Transverse field strength

h_field = 1