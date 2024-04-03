import numpy as np
import matplotlib.pyplot as plt

# Constants
h_bar = 1   # Reduced Planck constant (arbitrary units)
m = 1       # Particle mass (arbitrary units)
L = 1       # Width of the well (arbitrary units)
N = 100     # Number of grid points

# Discretization
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# Constructing the Hamiltonian matrix
H = np.zeros((N, N))
for i in range(1, N - 1):
    H[i, i] = -2 / dx**2 + 0  # Kinetic energy term (set to 0 for simplicity)
    H[i, i - 1] = 1 / dx**2
    H[i, i + 1] = 1 / dx**2

# Applying boundary conditions
H[0, 0] = 1 / dx**2  # Boundary condition at x = 0
H[N - 1, N - 1] = 1 / dx**2  # Boundary condition at x = L

# Solve the eigenvalue problem
E, psi = np.linalg.eigh(H)

# Plotting the results
plt.figure(figsize=(10, 6))
for n in range(5):  # Plot the first 5 eigenfunctions
    plt.plot(x, psi[:, n], label=f'Eigenfunction {n+1}')

plt.title('Wave Functions for a Particle in an Infinite Potential Well')
plt.xlabel('Position (x)')
plt.ylabel('Wave Function (ψ)')
plt.legend()
plt.grid(True)
plt.show()

# Print the first few eigenvalues
print("Eigenvalues (Energy levels):")
for n in range(5):
    print(f"E_{n+1} = {E[n]}")
