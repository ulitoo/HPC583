import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
G = 6.67430e-11  # gravitational constant, m^3 kg^-1 s^-2
m_sun = 1.989e30  # mass of the Sun, kg
m_earth = 5.972e24  # mass of the Earth, kg
r_earth = 1.496e11  # average distance from the Earth to the Sun, m
v_earth = 29500  # average orbital speed of the Earth, m/s
#v_earth = 29780  # average orbital speed of the Earth, m/s

# There is a PRECESION due to calculation errors, how to minimize? (reduce the steps when acceleration and velocity is big?)
# Solution: Runge Kutta to increase derivative precision
#           adaptative time step wrt with velocity!

# Initial conditions
x_earth = r_earth
y_earth = 0
vx_earth = 0
vy_earth = v_earth

# Time
dt_initial = 24 * 3600  # 1 day in seconds
total_days = 365  # Total simulation time (5 years)

# Trail parameters
trail_length = 150  # number of previous positions to keep for the trail
trail_x = []
trail_y = []

# Function to update Earth's position
def update_earth_position():
    global x_earth, y_earth, vx_earth, vy_earth, trail_x, trail_y
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

def acceleration(r):
    x, y = r
    r_mag = np.sqrt(x**2 + y**2)
    F_gravity = G * m_sun * m_earth / r_mag**2
    ax = -F_gravity * x / r_mag / m_earth
    ay = -F_gravity * y / r_mag / m_earth
    return np.array([ax, ay])

# Function to update Earth's position with adaptive time step
def update_earth_position_adaptive(dt):
    global x_earth, y_earth, vx_earth, vy_earth
    r = np.array([x_earth, y_earth])
    v = np.array([vx_earth, vy_earth])
    a = acceleration(r)
    # Estimate time step based on acceleration and velocity
    dt_new = dt_initial / (np.linalg.norm(v)/(v_earth))  # Adjust scaling factor as needed wrt velicuty

    # Update position using RK4 with the estimated time step
    k1r = v
    k1v = a
    k2r = v + 0.5 * dt_new * k1v
    k2v = acceleration(r + 0.5 * dt_new * k1r)
    k3r = v + 0.5 * dt_new * k2v
    k3v = acceleration(r + 0.5 * dt_new * k2r)
    k4r = v + dt_new * k3v
    k4v = acceleration(r + dt_new * k3r)
    x_earth += (dt_new / 6) * (k1r[0] + 2*k2r[0] + 2*k3r[0] + k4r[0])
    y_earth += (dt_new / 6) * (k1r[1] + 2*k2r[1] + 2*k3r[1] + k4r[1])
    vx_earth += (dt_new / 6) * (k1v[0] + 2*k2v[0] + 2*k3v[0] + k4v[0])
    vy_earth += (dt_new / 6) * (k1v[1] + 2*k2v[1] + 2*k3v[1] + k4v[1])
    trail_x.append(x_earth)
    trail_y.append(y_earth)
    if len(trail_x) > trail_length:
        trail_x.pop(0)
        trail_y.pop(0)
    return dt_new

# Function to update Earth's position using RK4 method - Runge-Kutta method 
def update_earth_position_rk4():
    global x_earth, y_earth, vx_earth, vy_earth
    h = dt
    r = np.array([x_earth, y_earth])
    v = np.array([vx_earth, vy_earth])
    k1r = v
    k1v = acceleration(r)
    k2r = v + 0.5 * h * k1v
    k2v = acceleration(r + 0.5 * h * k1r)
    k3r = v + 0.5 * h * k2v
    k3v = acceleration(r + 0.5 * h * k2r)
    k4r = v + h * k3v
    k4v = acceleration(r + h * k3r)
    x_earth += (h / 6) * (k1r[0] + 2*k2r[0] + 2*k3r[0] + k4r[0])
    y_earth += (h / 6) * (k1r[1] + 2*k2r[1] + 2*k3r[1] + k4r[1])
    vx_earth += (h / 6) * (k1v[0] + 2*k2v[0] + 2*k3v[0] + k4v[0])
    vy_earth += (h / 6) * (k1v[1] + 2*k2v[1] + 2*k3v[1] + k4v[1])
    trail_x.append(x_earth)
    trail_y.append(y_earth)
    if len(trail_x) > trail_length:
        trail_x.pop(0)
        trail_y.pop(0)

# Function to update animation
def update(frame):
    dt = update_earth_position_adaptive(dt_initial)  # Initial guess for time step
    #update_earth_position()
    #update_earth_position_rk4()
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