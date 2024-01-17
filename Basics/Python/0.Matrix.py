import sympy as sp

# Define a symbolic matrix
A = sp.Matrix([[2, -1, 1], [-3, -1, 4], [-1, 1, -1]])

# Perform LU decomposition
L, U, _ = A.LUdecomposition()

# Pretty print the matrices
sp.init_printing()
print("Matrix A:")
sp.pprint(A)

print("\nLower triangular matrix L:")
sp.pprint(L)

print("\nUpper triangular matrix U:")
sp.pprint(U)

# Verify the decomposition
verification = L * U

print("\nVerification (L * U):")
sp.pprint(verification.simplify())
