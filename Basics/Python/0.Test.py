import numpy as np

def compute_eigenvalues(N):
    # Constants
    m = 9.10938356e-31  # Mass of electron (kg)
    h_bar = 1.0545718e-34  # Reduced Planck's constant (Js)
    L = 5.2918e-11     # Bohr radius (m)

    constante = h_bar*h_bar/(2*m)
    
    # Conversion factor from Joules to eV
    joules_to_ev = 6.242e18
    

    # Constructing the Hamiltonian matrix
    a = -h_bar**2 / (2 * m)  # diagonal element
    #b = h_bar**2 / (m)  # off-diagonal element
    b = h_bar**2 / (m * L**2)  # off-diagonal element

    # Constructing the diagonal and off-diagonal arrays
    diagonal = a * np.ones(N)
    off_diagonal = b * np.ones(N - 1)

    # Constructing the Hamiltonian matrix
    H = np.diag(diagonal) + np.diag(off_diagonal, k=1) + np.diag(off_diagonal, k=-1)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)
    
    # Convert eigenvalues to eV
    eigenvalues_ev = eigenvalues * joules_to_ev

    return eigenvalues_ev

# Set the number of grid points
N = 10

# Compute eigenvalues
eigenvalues_ev = compute_eigenvalues(N)

# Print the computed eigenvalues in eV
print("Eigenvalues (eV):")
for i, eigenvalue_ev in enumerate(eigenvalues_ev):
    print(f"Eigenvalue {i + 1}: {eigenvalue_ev:.5e} eV")
