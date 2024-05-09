import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Constants
#h_bar = 1   # Reduced Planck constant (arbitrary units)
#m = 3       # Particle mass (arbitrary units)
#L = 2       # Width of the well (arbitrary units)
N = 100     # Number of grid points
m = 9.1094e-31     # Mass of electron (kg)
h_bar = 1.0546e-34  # Planck's constant over 2*pi (J.s)
e = 1.6022e-19     # Electron charge (C)
L = 5.2918e-11     # Bohr radius (m)

Hamiltonian_constant = h_bar*h_bar/(2*m)

# Analytical solution
def analytical_eigenvalues(n):
    return (n**2 * np.pi**2 * h_bar**2) / (2 * m * L**2)

# Numerical solution
def numerical_eigenvalues(N):
    # Discretization
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]

    # Constructing the Hamiltonian matrix
    H = np.zeros((N, N))
    for i in range(1, N - 1):
        H[i, i] = -2 *(Hamiltonian_constant)/ dx**2
        H[i, i - 1] = (Hamiltonian_constant) / dx**2
        H[i, i + 1] = (Hamiltonian_constant) / dx**2

    sp.pprint(H)
    plt.imshow(H, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    #plt.show()

    # Applying boundary conditions
    H[0, 0] = (Hamiltonian_constant) / dx**2
    H[N - 1, N - 1] = (Hamiltonian_constant) / dx**2

    # Solve the eigenvalue problem
    E, _ = np.linalg.eigh(H)
    return E - E[0]

# Compute analytical eigenvalues
n_values = np.arange(1, 6)  # Compute the first 5 eigenvalues
analytical_values = analytical_eigenvalues(n_values)/e

# Compute numerical eigenvalues
numerical_values = numerical_eigenvalues(N)/e

# Print and compare eigenvalues
print("Comparison of Eigenvalues:")
print(" n  | Analytical Energy | Numerical Energy | Relative Error")
print("-" * 100)
for i, n in enumerate(n_values):
    analytical = analytical_values[i]
    numerical = numerical_values[i]
    rel_error = np.abs((numerical - analytical) / analytical) * 100
    print(f"{n:2d}  | {analytical:17.5f}  | {numerical:17.5f}  | {rel_error:14.5f}%")
