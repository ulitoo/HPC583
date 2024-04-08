import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
G = 6.67430e-11  # gravitational constant, m^3 kg^-1 s^-2
m_sun = 1.989e30  # mass of the Sun, kg
m_earth = 5.972e24  # mass of the Earth, kg
r_earth = 1.496e11  # average distance from the Earth to the Sun, m
v_earth = 12080  # average orbital speed of the Earth, m/s
#v_earth = 29780  # average orbital speed of the Earth, m/s

# There is a PRECESION due to calculation errors, how to minimize? (reduce the steps when acceleration and velocity is big?)

# Initial conditions
x_earth = r_earth
y_earth = 0
vx_earth = 0
vy_earth = v_earth

# Time
dt = 24 * 3600  # 1 day in seconds
t = 0
total_days = 365  # Total simulation time (5 years)

# Trail parameters
trail_length = 150  # number of previous positions to keep for the trail
trail_x = []
trail_y = []

# Function to update Earth's position
def update_earth_position():
    global x_earth, y_earth, vx_earth, vy_earth#, trail_x, trail_y
    # Calculate gravitational force
    r = np.sqrt(x_earth**2 + y_earth**2)
    F_gravity = G * m_sun * m_earth / r**2
    # Calculate acceleration
    ax = -F_gravity * x_earth / r / m_earth
    ay = -F_gravity * y_earth / r / m_earth
    # Update velocity
    vx_earth += ax * dt
    vy_earth += ay * dt
    # Update position
    x_earth += vx_earth * dt
    y_earth += vy_earth * dt
    # Update trail
    trail_x.append(x_earth)
    trail_y.append(y_earth)
    if len(trail_x) > trail_length:
        trail_x.pop(0)
        trail_y.pop(0)

# Function to update animation
def update(frame):
    update_earth_position()
    trail.set_data(trail_x, trail_y)
    earth.set_data(x_earth, y_earth)
    return earth, trail

# Initialize plot
fig, ax = plt.subplots()
ax.set_xlim(-2*r_earth, 2*r_earth)
ax.set_ylim(-2*r_earth, 2*r_earth)
ax.set_aspect('equal', 'box')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Earth Orbiting Around the Sun')

# Plot the Sun
ax.plot(0, 0, 'ro', markersize=10, label='Sun')

# Plot the initial position of Earth
earth, = ax.plot([], [], 'bo', markersize=5, label='Earth')

# Plot the trail
trail, = ax.plot([], [], 'b--', alpha=0.3, label='Trail')

# Plot legend
ax.legend()

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(total_days), interval=0, blit=True)

# Show plot
plt.show()
