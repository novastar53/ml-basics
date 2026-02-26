import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Define the Lorenz System
def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# 2. Parameters
# We'll start two points ALMOST in the same spot (0.01% difference)
z0_1 = [1.0, 1.0, 1.0]
z0_2 = [1.0 + 1e-4, 1.0, 1.0] 

t_span = [0, 50]
t_eval = np.linspace(0, 50, 5000)

# Solve for both starting points
sol1 = solve_ivp(lorenz, t_span, z0_1, t_eval=t_eval)
sol2 = solve_ivp(lorenz, t_span, z0_2, t_eval=t_eval)

# 3. Visualization
fig = plt.figure(figsize=(12, 6))

# --- Plot 1: The 3D Attractor ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(sol1.y[0], sol1.y[1], sol1.y[2], 'b', lw=0.5, alpha=0.8)
ax1.set_title("The Lorenz Attractor (3D)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

# --- Plot 2: The Butterfly Effect (x vs time) ---
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(sol1.t, sol1.y[0], 'b', label='Start Point A', lw=1)
ax2.plot(sol2.t, sol2.y[0], 'r--', label='Start Point B (0.01% offset)', lw=1)
ax2.set_title("Sensitivity to Initial Conditions")
ax2.set_xlabel("Time")
ax2.set_ylabel("Value of x")
ax2.set_xlim(0, 40)
ax2.legend()

plt.tight_layout()
plt.show()
