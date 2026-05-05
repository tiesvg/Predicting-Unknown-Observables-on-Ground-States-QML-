'''
In this file, I estimate the expectation value of a certain observable in the ground state of the Hamiltonian
using the Hardware-efficient VQE
'''

import pennylane.numpy as np
import pennylane as qml


def SA_VQE_expec_val(dim_grid, hamiltonian, observable, depth, opt_steps, learning_rate):
    '''
    Execute the Variational Quantum Eigensolver (VQE) algorithm to find the ground state energy and 
    ground state expectation value of a certain observable (and Paulis that compose it)

    Inputs
    ------
    dim_grid: tuple
        Contains the number of rows and columns of the rectangular spin lattice

    hamiltonian: qml.Hamiltonian
        Hamiltonian representation of the system for specific coupling coefficients

    observable: qml.Hamiltonian
        Representation of the k-local observable whose ground state expectation value we want to evaluate

    depth: int
        Number of times that we apply the RZ,RX,RZ,CNOT block in VQE

    opt_steps: int
        Number of iterations in the optimization

    learning_rate: float
        Adam initial step size

    Outputs
    ---------
    obs_ground_state: float
        Estimate of the expectation value of the observable in the ground state

    Pauli_strings_ground_state: list 
        List of the ground state expectation values of the Pauli strings that compose the observable

    ground_state_energy: float
        Estimate of the expectation value of the Hamiltonian in the ground state
    '''

    # Generate the coordinates of the qubits in the 2D grid of dimension dim_grid
    n_qubits = dim_grid[0]*dim_grid[1]
    qubit_indices = [str((i,j)) for i in range(dim_grid[0]) for j in range(dim_grid[1])]

    # Select device
    dev = qml.device("default.qubit", wires=qubit_indices)
    n_params = n_qubits*3*depth

    # Define the hardware-efficient ansatz circuit
    def hea_ansatz(params):
        for d in range(depth):
            for q in range(n_qubits):
                x = q // dim_grid[1]
                y = q % dim_grid[1]
                wire = str((x,y))
                qml.RZ(params[3*d, q], wires=wire)
                qml.RX(params[3*d+1, q], wires=wire)
                qml.RZ(params[3*d+2, q], wires=wire)
                
            
            for i in range(0, n_qubits-1, 2):
                x_ctrl = i // dim_grid[1]
                y_ctrl = i % dim_grid[1]
                wire_ctrl = str((x_ctrl,y_ctrl))
                x_target = (i+1) // dim_grid[1]
                y_target = (i+1) % dim_grid[1]
                wire_target = str((x_target, y_target))
                qml.CNOT(wires=[wire_ctrl, wire_target])


    # Measure of the expectation value of the hamiltonian (energy)
    @qml.qnode(dev)
    def energy(params):
        hea_ansatz(params)
        return qml.expval(hamiltonian)
    
    # Measure of the expectation value of the input observable in the estimated ground state
    @qml.qnode(dev)
    def observable_exp_val(params):
        hea_ansatz(params)
        return qml.expval(observable)
    
    # Measure of the expectation value of an input Pauli string (that composes the observable)
    @qml.qnode(dev)
    def pauli_string_exp_val(params, pauli_string):
        hea_ansatz(params)
        return qml.expval(pauli_string)


    def optimizer(params):
        '''
        Optimizer to adjust the parameters. We use Adam optimizer

        Input
        -------
        params: np.ndarray
            An array with the trainable parameters of the QAOA ansatz.
        
        Output
        -------
        params: np.ndarray
            An array with the optimized parameters of the HEA ansatz.
        ''' 
    
        opt = qml.AdamOptimizer(learning_rate)

        for _ in range(opt_steps):
            params = opt.step(energy, params)
        
        return params


    # Initalize parameters randomly using a uniform distribution
    params_initial = np.random.uniform(-np.pi, np.pi, size=(3*depth, n_qubits))
    
    # Obtain the final parameters after the whole optimization loop
    params_final = optimizer(params_initial)

    # Evaluate the estimation of the ground state energy
    ground_state_energy = energy(params_final)

    # Evaluate the expectation value of the observable
    obs_ground_state = observable_exp_val(params_final)

    # Evaluate the ground state expectation values of all the Pauli strings contained in the observable passes as an input
    Pauli_strings_ground_state = []
    for pauli_string in observable.ops[0]:
        Pauli_strings_ground_state.append(pauli_string_exp_val(params_final, pauli_string).item())
    

    return obs_ground_state.item(), Pauli_strings_ground_state, ground_state_energy.item()
