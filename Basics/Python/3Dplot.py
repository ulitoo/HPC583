from sympy import symbols, cos, sin
from sympy.plotting import plot3d

# Define variables
x, y = symbols('x y')
max = 1.5

# Define a function in terms of x and y
z = cos(x**2 + y**2)

# Create a 3D plot
plot3d(z, (x, -max, max), (y, -max, max))
