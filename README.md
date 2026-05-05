# Supervised Learning for Quantum Many-Body Ground States

A comparison of supervised learning methods for predicting ground state expectation values of unknown observables for a class of local Hamiltonians. Three approaches are benchmarked against each other: a quantum LASSO regression that calls VQE, a classical LASSO regression with a random Fourier feature map, and a deep neural network.

## Background

Computing ground state properties of many-body Hamiltonians is hard in general. An alternative is to learn the map from Hamiltonian parameters to observable expectation values from data, so that predictions on new instances become cheap. This project trains and compares three such models, and checks how well their estimates match the exact values obtained by diagonalization.

## What's in here

- `main.py` — entry point. Set `mode = 'A'` to compare estimated ground state energies and observables against exact diagonalization, or `mode = 'B'` to train and benchmark the three models against each other.
- `parameters.py` — lattice size, number of training examples, and model hyperparameters.
- `SA_VQE.py` — VQE-related functions. *Despite the name, this uses the hardware-efficient ansatz, not the symmetry-adapted one — the file was never renamed.*
- `fourier_feature_map.py` — the classical LASSO model based on a random Fourier feature map (Ref. 3).
- `functions.py` — coupling generation, Hamiltonian and observable construction, exact diagonalization via numpy, training dataset generation, and the ML model definitions.

## Requirements

`numpy`, `matplotlib`, `pennylane`, `networkx`, `sklearn`, `tensorflow`
