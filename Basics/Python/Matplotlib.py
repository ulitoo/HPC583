import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(0, 2 * np.pi, 100)

# Calculate sine and cosine values
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the sine function
plt.plot(x, y_sin, label='sin(x)')

# Plot the cosine function
plt.plot(x, y_cos, label='cos(x)')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine and Cosine Functions')
plt.legend()

# Show the plot
plt.show()
