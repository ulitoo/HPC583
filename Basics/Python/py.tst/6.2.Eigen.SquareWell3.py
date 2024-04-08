import numpy as np

# Constants
m = 9.1094e-31     # Mass of electron (kg)
hbar = 1.0546e-34  # Planck's constant over 2*pi (J.s)
e = 1.6022e-19     # Electron charge (C)
L = 5.2918e-11     # Bohr radius (m)
N = 1000         # Number of grid points


# Calculate step size
h = L / (N - 1)

# Potential function
def V(x):
    return 0.0

def f(r, x, E):
    psi, phi = r
    fpsi = phi
    fphi = (2*m/hbar**2) * (V(x)-E) * psi
    return np.array([fpsi, fphi], float)

# Calculate the wavefunction for a particular energy
def solve(E):
    psi = 0.0
    phi = 1.0
    r = np.array([psi, phi], float)

    for x in np.linspace(0, L, N):
        k1 = np.array([h * f(r, x, E)[i] for i in range(2)])
        k2 = np.array([h * f(r + 0.5 * k1, x + 0.5 * h, E)[i] for i in range(2)])
        k3 = np.array([h * f(r + 0.5 * k2, x + 0.5 * h, E)[i] for i in range(2)])
        k4 = np.array([h * f(r + k3, x + h, E)[i] for i in range(2)])
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return r[0]

# Main program to find the ground state energy using the secant method
E1 = 0.0
E2 = e
psi2 = solve(E1)

target = e / 1000
while abs(E1 - E2) > target:
    psi1, psi2 = psi2, solve(E2)
    E1, E2 = E2, E2 - psi2 * (E2 - E1) / (psi2 - psi1)

# Convert the energy eigenvalue from J to eV
ground_energy = E2 / e

print("Ground state energy:", ground_energy, "eV")
