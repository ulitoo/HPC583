import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initial conditions
x1_0, y1_0 = -1, 0  # initial position of body 1
v1x_0, v1y_0 = 0, 1  # initial velocity of body 1

x2_0, y2_0 = 1, 0  # initial position of body 2
v2x_0, v2y_0 = 0, -1  # initial velocity of body 2

# Function to update positions
def update(frame):
    global x1, y1, x2, y2

    x1 += v1x_0
    y1 += v1y_0
    x2 += v2x_0
    y2 += v2y_0

    line.set_data([x1, x2], [y1, y2])
    return line,

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)

# Set up the line
line, = ax.plot([], [], 'o--', lw=1)

# Initialize the positions
x1, y1 = x1_0, y1_0
x2, y2 = x2_0, y2_0

# Create animation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

plt.show()
