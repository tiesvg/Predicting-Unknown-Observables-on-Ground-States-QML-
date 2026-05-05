'''
In this file, I will define subroutines use to generate random couplings, solve the ground state by numpy diagonalization, generate the training set,
perform LASSO regression and implement a custom neural network. Basically, all functions except the ones involving VQE and random Fourier feature map.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pennylane as qml
import networkx as nx
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf


from SA_VQE import SA_VQE_expec_val
import fourier_feature_map


def generate_couplings(dim_grid, seed=None):
    '''
    Generate the horizontal and vertical couplings between the qubits in the square lattice sampling from a uniform distribution {-1,1}

    Inputs
    ------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

    rng: numpy.random.mtrand.RandomState
        Random number generator (for reproducible results)

    Outputs
    -------
    J_right: np.ndarray
        Horizontal couplings
        Dimension dim_grid[0] x (dim_grid[1]-1)

    J_down: np.ndarray
        Vertical couplings
        Dimension (dim_grid[0]-1) x dim_grid[1]
    '''

    if seed is not None:
        np.random.seed(seed)
    

    J_right = np.random.uniform(-1,1, size=(dim_grid[0], dim_grid[1]-1))
    J_down = np.random.uniform(-1,1, size=(dim_grid[0]-1, dim_grid[1]))

    return J_right, J_down



def visualize_couplings(dim_grid, J_right, J_down):
    '''
    Visualizes the couplings between spins in a 2D grid, with values ranging from -1 to 1.

    Inputs
    -------
    dim_grid: tuple
        Number of rows and columns (rows, columns)

    J_right: np.ndarray
        Horizontal couplings, shape (rows, columns - 1)

    J_down: np.ndarray
        Vertical couplings, shape (rows - 1, columns)
    '''

    rows, columns = dim_grid
    G = nx.Graph()
    pos = {}

    # Create color map (blue to white to red)
    cmap = cm.seismic
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    # Add nodes and positions
    for i in range(rows):
        for j in range(columns):
            qubit = (i, j)
            G.add_node(qubit)
            pos[qubit] = (j, -i)

    # Add horizontal edges with weight/color
    for i in range(rows):
        for j in range(columns - 1):
            q1 = (i, j)
            q2 = (i, j + 1)
            value = J_right[i, j]
            G.add_edge(q1, q2, weight=value, color=cmap(norm(value)))

    # Add vertical edges with weight/color
    for i in range(rows - 1):
        for j in range(columns):
            q1 = (i, j)
            q2 = (i + 1, j)
            value = J_down[i, j]
            G.add_edge(q1, q2, weight=value, color=cmap(norm(value)))

    # Extract edge attributes
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    edge_widths = [2 for _ in G.edges()]  # Optional: could scale with abs(weight)

    # Create figure and axis
    fig, ax = plt.subplots()

    # Draw graph
    nx.draw(
        G, pos, with_labels=True, node_color='lightgray',
        edge_color=edge_colors, width=edge_widths, node_size=700, ax=ax
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Coupling Strength (J)")

    ax.set_title("2D Spin Grid with Coupling Values (−1 to +1)")
    ax.axis('off')
    plt.show()



def hamiltonian(dim_grid, J_right, J_down, hamiltonian_label):
    '''
    Computes the Hamiltonian of a 2D antiferromagnetic system of N spins

    Inputs
    ------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

     J_right: np.ndarray
        Horizontal couplings
        Dimension n x (n-1)

    J_down: np.ndarray
        Vertical couplings
        Dimension (n-1) x n

    hamiltonian_label: str
        Identifier of the Hamiltonian: 'heisenberg', 'ising'
    
    Output
    ------
    hamiltonian: qml.Hamiltonian
        Hamiltonian representation of the system for specific coupling coefficients
        
    '''

    coef = []
    ops = []

    if hamiltonian_label == 'heisenberg':
        # Fill the horizontal interactions
        for i in range(dim_grid[0]):
            for j in range(dim_grid[1]-1):
                index1 = str((i,j))
                index2 = str((i,j+1))
                coef.append(J_right[i,j])
                ops.append(qml.PauliX(index1)@qml.PauliX(index2) + qml.PauliY(index1)@qml.PauliY(index2) + qml.PauliZ(index1)@qml.PauliZ(index2))
                #ops.append(qml.PauliZ(index1)@qml.PauliZ(index2))

        # Fill the vertical interactions
        for i in range(dim_grid[0]-1):
            for j in range(dim_grid[1]):
                index1 = str((i,j))
                index2 = str((i+1,j))
                coef.append(J_down[i,j])
                ops.append(qml.PauliX(index1)@qml.PauliX(index2) + qml.PauliY(index1)@qml.PauliY(index2) + qml.PauliZ(index1)@qml.PauliZ(index2))
                #ops.append(qml.PauliZ(index1)@qml.PauliZ(index2))

    elif hamiltonian_label == 'ising':
        # Fill the horizontal interactions
        for i in range(dim_grid[0]):
            for j in range(dim_grid[1]-1):
                index1 = str((i,j))
                index2 = str((i,j+1))
                coef.append(J_right[i,j])
                ops.append(qml.PauliZ(index1)@qml.PauliZ(index2))

        # Fill the vertical interactions
        for i in range(dim_grid[0]-1):
            for j in range(dim_grid[1]):
                index1 = str((i,j))
                index2 = str((i+1,j))
                coef.append(J_down[i,j])
                ops.append(qml.PauliZ(index1)@qml.PauliZ(index2))


    hamiltonian = qml.Hamiltonian(coef, ops)

    return hamiltonian

def observable(dim_grid, obs_name, qubits):
    '''
    Computes the Hamiltonian of a 2D antiferromagnetic system of N spins

    Inputs
    ------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

    obs_name: str
        Name/identification of the observable to measure

    qubits: tuple
        Tuple containing the tuples representing the x and y coordinates of the qubits in the lattice that we want to measure


    Output
    ------
    observable: qml.Hamiltonian
        Representation of the observable that we want to measure
        
    '''

    if obs_name == 'correlation':
        # Define the observable to measure: the correlation function between qubits 0 and 1
        coefs = [1/3]
        i0 = str(qubits[0])
        i1 = str(qubits[1])
        ops = [qml.PauliX(i0) @ qml.PauliX(i1) + qml.PauliY(i0) @ qml.PauliY(i1) + qml.PauliZ(i0) @ qml.PauliZ(i1)]
        observable = qml.Hamiltonian(coefs, ops)

    return observable


def ground_state_expectation_value(dim_grid, hamiltonian, observable):
    '''
    Computes the ground state expectation value of a given observable classically

    Inputs
    ------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

    hamiltonian: qml.Hamiltonian
        Hamiltonian representation of the system for specific coupling coefficients

    observable: qml.Hamiltonian
        Representation of the k-local observable whose ground state expectation value we want to evaluate

    Output
    ------
    expectation_value_obs: float
        Formula: Tr(\rho(x) O)

    expectation_value_Paulis: list
        List of the graound state expectation value of the Pauli strings in which the observable is decomposed in

    ground_state_energy: float
        Ground state energy of the given Hamiltonian
        
    '''
     
    # Transform the qml.Hamiltonian objects to np matrices
    wires = [str((i, j)) for i in range(dim_grid[0]) for j in range(dim_grid[1])]
    H_matrix = qml.matrix(hamiltonian, wire_order=wires)
    obs_matrix = qml.matrix(observable, wire_order=wires)

    # Diagonalize the Hamiltonian to find the ground state
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    ground_state_energy = eigenvalues[0]
    ground_state = eigenvectors[:,0]

    # Calculate the expectation value of correlation_01 in the ground state
    expectation_value_obs = (ground_state.conj().T @ obs_matrix @ ground_state).real

    # Calculate the expectation value of the Pauli strings in which the observable is decomposed
    expectation_value_Paulis = []
    Pauli_strings = observable.ops[0]

    for pauli_string in Pauli_strings:
        # Transform the qml.Hamiltonian Pauli string object to an np matrix
        pauli_string_matrix = qml.matrix(pauli_string, wire_order=wires)
        # Calculate the expectation value of correlation_01 in the ground state
        expectation_value_Paulis.append((ground_state.conj().T @ pauli_string_matrix @ ground_state).real)

    return expectation_value_obs, np.array(expectation_value_Paulis), ground_state_energy



def generate_training_set(dim_grid, hamiltonian_label, num_examples, observable_name, qubits, depth=None, opt_steps=None, learning_rate=None, delta=None, gamma=None, R=None):
    '''
    Generate the training set for the model, including feature map and output (ground state expectation value of observable)

    Inputs
    ------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice
    
    hamiltonian_label: str
        Identifier of the Hamiltonian: 'heisenberg', 'ising'

    num_examples: int
       Number of training examples |{x^l={J_{ij}^l, y^l}|

    observable_name: str
        Name/identification of the k-local observable whose ground state expectation value we want to evaluate

    qubits: tuple of 2 tuples
        Tuple of the coordinates of the two qubits where the correlation observable acts on
    
    depth: int
        Depth of the hardware-efficient VQE

    opt_steps: int
        Number of optimization steps for VQE

    learning_rate: float
        Learning rate/gradient descent step size for Adam optimizer in VQE

    delta: int
        Maximum Manhattan distance for selecting edges close to the region where the observable acts on

    gamma: float
        Hyperparameter (related to the frequency) of the random Fourier feature map
    
    R: int
        Hyperparameter (related to dimension) of the random Fourier feature map
    
    
    Outputs
    -------
    X: np.ndarray
        Matrix of dimension num_examples x J_len containing num_examples of the input vector
    
    PhiX_quantum_diag: np.ndarray
        Matrix (of dimension num_examples x number of Pauli strings in which the observable can be decomposed in) of the feature map for the quantum model
        Each element is tr(rho(x)P_i), i.e., expectation value of a Pauli
        Obtained by diagonalization using numpy

    PhiX_quantum_VQE: np.ndarray
        Matrix (of dimension num_examples x number of Pauli strings in which the observable can be decomposed in) of the feature map for the quantum model
        Each element is tr(rho(x)P_i), i.e., expectation value of a Pauli
        Obtained by VQE

    PhiX_fourier: np.ndarray
        Matrix {of dimension num_examples x 2R(number of local couplings)} associated with random Fourier feature map
    
    Y_diag: np.ndarray
        Vector of length num_examples with the expectation values of the measured observable
        Obtained by diagonalization with numpy

    Y_VQE: np.ndarray
        Vector of length num_examples with the expectation values of the measured observable
        Obtained by VQE
        
    ''' 
    
    # Obtain observable from its name and the qubits it acts on
    obs = observable(dim_grid, observable_name, qubits)

    # Initialize the inputs and ouputs of the ML model
    J_len = int(2*dim_grid[0]*dim_grid[1] - dim_grid[0] - dim_grid[1])      # Number of couplings = len(J_right) + len(J_down)
    X = np.zeros((num_examples, J_len))                                     # Matrix of dimension: num_examples x number of couplings (J_{ij} flattened)
    Y_diag = np.zeros(num_examples)                                         # Column with the ground state expectation value of obs (obtained with diagonalization)
    Y_VQE = np.zeros(num_examples)                                          # Column with the ground state expectation value of obs (obtained with VQE)

    # Initialize Quantum Feature Map
    PhiX_quantum_diag = np.zeros((num_examples, len(obs.ops[0])))           # Matrix of dimension: num_examples x number of Pauli strings in obs (by diagonalization)
    PhiX_quantum_VQE = np.zeros((num_examples, len(obs.ops[0])))            # Matrix of dimension: num_examples x number of Pauli strings in obs (by VQE)
    
    # Initialize Random Fourier Feature Map
    local_right_edges, local_down_edges = fourier_feature_map.get_local_edges(qubits, dim_grid, delta)      # Binary strings encoding the local edges
    l = np.sum(local_down_edges) + np.sum(local_down_edges)
    w = np.random.normal(0, 1, R)                                                                           # Random vector of weights for Fourier feature map
    PhiX_fourier = np.zeros((num_examples, ((np.sum(local_right_edges) + np.sum(local_down_edges))*2*R)))   # Matrix of dimension num_examples x (number of local edges)*2R
    
    
    # Generate feature maps and outputs for each examples
    for l in range(num_examples):
        # Generate random couplings, flatten them out and concatenate them to generate the input of the ML model
        J_right, J_down = generate_couplings(dim_grid)
        J_right_flat = J_right.flatten()
        J_down_flat = J_down.flatten()
        X[l,:] = np.concatenate((J_right_flat, J_down_flat))

        # Obtain output and Quantum Feature map by diagonalization and VQE
        Y_diag[l], PhiX_quantum_diag[l,:], _ = ground_state_expectation_value(dim_grid, hamiltonian(dim_grid, J_right, J_down, hamiltonian_label), obs)
        Y_VQE[l], PhiX_quantum_VQE[l,:], _ = SA_VQE_expec_val(dim_grid, hamiltonian(dim_grid, J_right, J_down, hamiltonian_label), obs, depth, opt_steps, learning_rate)

        # Obtain the random Fourier feature map
        Z = fourier_feature_map.generate_local_couplings(J_right, J_down, local_right_edges, local_down_edges)
        PhiX_fourier[l,:] = fourier_feature_map.random_fourier_feature_map(Z, w, gamma, R)


    return X, PhiX_quantum_diag, PhiX_quantum_VQE, PhiX_fourier, Y_diag, Y_VQE



def lasso_regression(X, y):
    '''
    Perform Cross-validation LASSO, trying different hyperparamers (alphas) that represent the importance of the regularization

    Inputs
    -------
    X: np.ndarray
        Matrix of dimension num_examples x number of features
    
    Y: np.ndarray
        Vector of length num_examples

    Outputs
    -------
    coefficients: np.ndarray
        Represent the importance of each feature for the prediction of the output
    
    L1_norm_coef: float
        L1 norm of the vector of the coefficients

    optimal_alpha: float
        Value of alpha that minimizes the cross validation loss

    train_mse: float
        Mean squared error in training data

    test_mse: float
        Mean squared error in test data

    train_r2: float
        R2 score of training data

    test_r2: float
        R2 score of test data
    '''

    # Split the dataset in training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define an object of the class Lasso and train the model
    lasso = LassoCV(alphas=[1e-4,1e-3,1e-2,1e-1,1], cv=5, random_state=42)
    lasso.fit(X_train, y_train)
    optimal_alpha = lasso.alpha_
    coefficients = lasso.coef_
    L1_norm_coef = np.linalg.norm(coefficients, ord=1)

    # Calculate the predictions for the test set
    y_pred_train = lasso.predict(X_train)
    y_pred_test = lasso.predict(X_test)

    # Evaluate the error and R^2 score in training data and test
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    return coefficients, L1_norm_coef, optimal_alpha, train_mse, test_mse, train_r2, test_r2


def neural_network(X, y):
    '''
    Trains a feedforward neural network on the provided dataset, with regularization,
    early stopping, and plots of training vs validation loss.

    Inputs
    -------
    X : np.ndarray
        Input features (perhaps after performing a feature map)
    y : np.ndarray
        Target values

    Outputs
    --------
    r2: float
        R^2 score (typically used to evaluate how good a regression model explains the data)
    
    mse: float
        Mean squared error in the test data
    '''

    # Split dataset: 80% train, 10% val, 10% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    dim_input = X.shape[1]
    num_examples = X_train.shape[0]

    # Set seed for reproducibility
    seed = 42
    tf.random.set_seed(seed)

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(dim_input,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation=None)
    ])

    # Compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(optimizer=opt, loss='mse')

    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=max(1, int(num_examples / 10)),
        epochs=100,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluate on test data
    output = model.predict(X_test)
    r2 = r2_score(y_test, output)
    mse = mean_squared_error(y_test, output)
    print(f"r2 score on test data = {r2:.4f}")
    print(f"Mean squared error on test data = {mse:.4f}")

    # Plot training vs validation loss
    plt.figure(figsize=(7, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return r2, mse
    