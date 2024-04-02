import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['animation.html'] = 'html5'

from matplotlib.animation import FuncAnimation

# Constants
G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)
m1 = 5.972e27     # mass of body 1 (kg)
m2 = 7.34767309e22 # mass of body 2 (kg)

# Initial conditions
x1_0, y1_0 = 0, 0.    # initial position of body 1 (m)
v1x_0, v1y_0 = 0., 0. # initial velocity of body 1 (m/s)

x2_0, y2_0 = 3.844e8, 0.    # initial position of body 2 (m)
v2x_0, v2y_0 = 0., 0.    # initial velocity of body 2 (m/s)

# Function to calculate acceleration due to gravity
def calculate_acceleration(x1, y1, x2, y2):
    r = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    a1 = -G * m2 / r**2
    a2 = -G * m1 / r**2
    theta = np.arctan2(y2 - y1, x2 - x1)
    ax1 = a1 * np.cos(theta)
    ay1 = a1 * np.sin(theta)
    ax2 = a2 * np.cos(theta)
    ay2 = a2 * np.sin(theta)
    return ax1, ay1, ax2, ay2

# Function to update positions
def update(frame):
    global x1, y1, v1x, v1y, x2, y2, v2x, v2y

    ax1, ay1, ax2, ay2 = calculate_acceleration(x1, y1, x2, y2)
    v1x += ax1 * dt
    v1y += ay1 * dt
    v2x += ax2 * dt
    v2y += ay2 * dt

    x1 += v1x * dt
    y1 += v1y * dt
    x2 += v2x * dt
    y2 += v2y * dt

    line.set_data([x1, x2], [y1, y2])
    return line,

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-2e11, 2e11)
ax.set_ylim(-2e11, 2e11)

# Set up the line
line, = ax.plot([], [], 'o-', lw=2)

# Initialize the positions and velocities
x1, y1 = x1_0, y1_0
v1x, v1y = v1x_0, v1y_0
x2, y2 = x2_0, y2_0
v2x, v2y = v2x_0, v2y_0

# Time parameters
dt = 3600  # time step (s)
num_frames = 1000

# Create animation
ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

plt.show()
