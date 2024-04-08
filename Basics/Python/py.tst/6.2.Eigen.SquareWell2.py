import numpy as np

# Constants
m = 9.1094e-31     # Mass of electron (kg)
hbar = 1.0546e-34  # Planck's constant over 2*pi (J.s)
e = 1.6022e-19     # Electron charge (C)
L = 5.2918e-11     # Bohr radius (m)
N = 100            # Number of grid points

# Discretization
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# Constructing the Hamiltonian matrix
H = np.zeros((N, N))
for i in range(1, N - 1):
    H[i, i] = -2 / dx**2
    H[i, i - 1] = 1 / dx**2
    H[i, i + 1] = 1 / dx**2

# Applying boundary conditions
H[0, 0] = 1 / dx**2
H[N - 1, N - 1] = 1 / dx**2

# Solve the eigenvalue problem
E, psi = np.linalg.eigh(H)

# Extract ground state energy
ground_energy = E[0] * (hbar**2 / (2 * m * L**2)) / e

print("Ground state energy:", ground_energy, "eV")
