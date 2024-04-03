import numpy as np
import matplotlib.pyplot as plt

# Define gravitational constant (in m^3/kg/s^2)
G = 6.67430e-11

# Define initial conditions
semi_major_axis = 1.5e11  # meters
eccentricity = 0.8  # Example eccentricity for demonstration
orbital_period = np.sqrt(4 * np.pi**2 * semi_major_axis**3 / (G * 1.989e30))  # seconds

# Calculate positions of foci
c = semi_major_axis * eccentricity
focus1 = (-c, 0)
focus2 = (c, 0)

# Define time points at which to calculate positions (e.g., one orbit)
num_points = 100
times = np.linspace(0, orbital_period, num_points)

# Calculate positions at each time point
positions = []
for t in times:
    mean_anomaly = 2 * np.pi * t / orbital_period
    eccentric_anomaly = mean_anomaly
    true_anomaly = 2 * np.arctan(np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(eccentric_anomaly / 2))
    radius = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(true_anomaly))
    x = radius * np.cos(true_anomaly)
    y = radius * np.sin(true_anomaly)
    positions.append((x, y))

# Extract x and y coordinates for plotting
x_coords, y_coords = zip(*positions)

# Calculate the center of the ellipse
center_x = (min(x_coords) + max(x_coords)) / 2
center_y = (min(y_coords) + max(y_coords)) / 2

# Center the orbit and foci around (0,0)
x_coords_centered = np.array(x_coords) - center_x
y_coords_centered = np.array(y_coords) - center_y

# Plot positions and foci
plt.figure(figsize=(8, 6))
plt.plot(x_coords_centered, y_coords_centered, marker='.', linestyle='', label='Orbit')
plt.plot(focus1[0], focus1[1], marker='o', color='red', label='Focus 1')
plt.plot(focus2[0], focus2[1], marker='o', color='green', label='Focus 2')
plt.title('Orbit of a Body in a Two-Body System with Foci (Centered)')
plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.legend()
plt.show()
