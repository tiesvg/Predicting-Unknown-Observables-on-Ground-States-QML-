'''
Author: Cesar Hernando

In this file, we train different Machine Learning (ML) models to learn to predict properties of a 2D antiferromagnetic system.

'''

import numpy as np
import matplotlib.pyplot as plt

import functions
from SA_VQE import SA_VQE_expec_val
import parameters

# Select the mode of the program: 
# A) basic test to test the performance of VQE and generate correlation matrix 
# B) Generate training data set (including performing feature map (both quantum and classical model) and train (and test) different ML models

mode = 'B'

if mode == 'A': # Quick testing
    # Set the number of qubits in each row/column of the square grid
    dim_grid = (2,2)
    num_qubits = dim_grid[0]*dim_grid[1]
    hamiltonian_label = 'heisenberg'

    # Generate the coupling coefficients
    seed = 3
    J_right, J_down = functions.generate_couplings(dim_grid, seed)
    functions.visualize_couplings(dim_grid, J_right, J_down)

    # Obtain the Hamiltonian (qml.Hamiltonian) of the 2D Antiferromagnetic lattice
    hamiltonian = functions.hamiltonian(dim_grid, J_right, J_down, hamiltonian_label)

    # Define the observable by its name and the qubits it affects
    # Options: obs_name = 'correlation', qubits = (q0, q1)
    obs_name = 'correlation'

    correlations_matrix_diag = np.zeros((num_qubits, num_qubits))
    correlations_matrix_SA_VQE = np.zeros((num_qubits, num_qubits))

    qubits = [(i,j) for i in range(dim_grid[0]) for j in range(dim_grid[1])]
    
    ground_state_energies_VQE = []

    for i, q0 in enumerate(qubits):
        for j, q1 in enumerate(qubits[i+1:], start=i+1):
    
            obs = functions.observable(dim_grid, obs_name, (q0, q1))

            # Obtain ground state property by diagonalization
            obs_diag, ground_state_energy_diag, ground_state_energy_diag = functions.ground_state_expectation_value(dim_grid, hamiltonian, obs)
            correlations_matrix_diag[i, j] = obs_diag
            correlations_matrix_diag[j, i] = obs_diag
            
            # Obtain ground state property by SA VQE
            depth = 3
            opt_steps = 500
            learning_rate = 0.01
            obs_SA_VQE, _, ground_state_energy_VQE = SA_VQE_expec_val(dim_grid, hamiltonian, obs, depth, opt_steps, learning_rate)
            ground_state_energies_VQE.append(ground_state_energy_VQE)
            correlations_matrix_SA_VQE[i, j] = obs_SA_VQE
            correlations_matrix_SA_VQE[j, i] = obs_SA_VQE
    
    # Plot the heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    im1 = axs[0].imshow(correlations_matrix_diag, cmap='viridis', vmin=np.min(correlations_matrix_diag), vmax=np.max(correlations_matrix_diag))
    axs[0].set_title("Correlation (Diagonalization)")
    plt.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(correlations_matrix_SA_VQE, cmap='viridis', vmin=np.min(correlations_matrix_SA_VQE), vmax=np.max(correlations_matrix_SA_VQE))
    axs[1].set_title("Correlation (SA VQE)")
    plt.colorbar(im2, ax=axs[1])

    for ax in axs:
        ax.set_xlabel("Qubit index")
        ax.set_ylabel("Qubit index")
        ax.set_xticks(range(num_qubits))
        ax.set_yticks(range(num_qubits))

    plt.tight_layout()
    plt.show()

    # Print the ground state energy obtained by diagonalization with numpy and with VQE
    print("Ground state energy [Diagonalization] = ", ground_state_energy_diag)
    print("Ground state energies [VQE] = ", ground_state_energies_VQE)
    print(f"Average = {np.mean(ground_state_energies_VQE)}, Standard deviation = {np.std(ground_state_energies_VQE)}")
  

elif mode == 'B': # Generating dataset and training ML models

    ##########################################################################################################
    ##### Definition of parameters and generation of the data using diagonalization (with numpy) and VQE #####
    ##########################################################################################################

    # Retrieve parameters from parameters file

    # First, general parameters
    dim_grid, obs_name, qubits, hamiltonian_label = parameters.dim_grid, parameters.obs_name, parameters.qubits, parameters.hamiltonian_label

    # Then, Training parameters
    num_examples = parameters.num_examples

    # Next, VQE parameters
    depth, opt_steps, learning_rate = parameters.delta, parameters.opt_steps, parameters.learning_rate
    
    # Finally, Random fourier map parameters
    delta, gamma, R = parameters.delta, parameters.gamma, parameters.R
    

    # Generate datasets and perform feature mapping of the inputs
    X, PhiX_quantum_diag, PhiX_quantum_VQE, PhiX_fourier, Y_diag, Y_VQE = functions.generate_training_set(dim_grid, hamiltonian_label, num_examples, obs_name, qubits, depth=depth, opt_steps=opt_steps, learning_rate=learning_rate, delta=delta, gamma = gamma, R=R)
    print("\nData set generated")
    
    #######################################################
    ##### Training and testing of different ML models #####
    #######################################################

    # 1. Neural network using data obtained by diagonalization

    print('\nNeural Network [output obtained by diagonalization]:\n')
    r2_nn_diag, mse_nn_diag = functions.neural_network(X, Y_diag)
    
    # 2. LASSO regression with quantum feature map using data obtained by VQE
    
    coefficients, L1_norm_coef, optimal_alpha, train_mse, test_mse, train_r2, test_r2 = functions.lasso_regression(PhiX_quantum_VQE, Y_VQE)
    print('\nLASSO with quantum feature map')
    print(f'Coefficients (omega)= {coefficients}')
    print(f'L1 norm of omega = {L1_norm_coef}')
    print(f"Optimal alpha: {optimal_alpha}")
    print(f'MSE: training -> {train_mse}; test -> {test_mse}')
    print(f'R^2: training -> {train_r2}; test -> {test_r2}')
    
    # 3. LASSO regression with random fourier feature map using data obtained by diagonalization

    coefficients, L1_norm_coef, optimal_alpha, train_mse, test_mse, train_r2, test_r2 = functions.lasso_regression(PhiX_fourier, Y_diag)
    print('\nLASSO with random Fourier feature map')
    print(f'Coefficients (omega)= {coefficients}')
    print(f'L1 norm of omega = {L1_norm_coef}')
    print(f"Optimal alpha: {optimal_alpha}")
    print(f'MSE: training -> {train_mse}; test -> {test_mse}')
    print(f'R^2: training -> {train_r2}; test -> {test_r2}')


   
    



