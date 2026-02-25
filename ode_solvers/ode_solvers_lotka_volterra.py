import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Define the System (The Derivative of the Vector [x, y])
def lotka_volterra(t, z, a=1.1, b=0.4, d=0.1, g=0.4):
    x, y = z
    dxdt = a*x - b*x*y
    dydt = d*x*y - g*y
    return [dxdt, dydt]

# 2. Parameters
z0 = [10, 5]     # Initial [Rabbits, Foxes]
t0, tf = 0, 100  # Longer time to see the cycles
n_steps = 1000   # More steps for a smoother curve
t_points = np.linspace(t0, tf, n_steps)
h = t_points[1] - t_points[0]

# --- METHOD A: Euler's Method ---
z_euler = np.zeros((n_steps, 2))
z_euler[0] = z0
for i in range(n_steps - 1):
    # Vectorized update: z_next = z_now + [dxdt, dydt] * dt
    slopes = lotka_volterra(t_points[i], z_euler[i])
    z_euler[i+1] = z_euler[i] + np.array(slopes) * h

# --- METHOD B: Runge-Kutta (RK4) ---
z_rk4 = np.zeros((n_steps, 2))
z_rk4[0] = z0
for i in range(n_steps - 1):
    ti = t_points[i]
    zi = z_rk4[i]
    
    k1 = np.array(lotka_volterra(ti, zi))
    k2 = np.array(lotka_volterra(ti + h/2, zi + h*k1/2))
    k3 = np.array(lotka_volterra(ti + h/2, zi + h*k2/2))
    k4 = np.array(lotka_volterra(ti + h, zi + h*k3))
    
    z_rk4[i+1] = zi + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# --- METHOD C: SciPy (solve_ivp) ---
sol = solve_ivp(lotka_volterra, [t0, tf], z0, t_eval=t_points)
z_scipy = sol.y.T # Transpose to match our (steps, 2) shape

# 3. Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- PLOT 1: Populations over Time (The Lag Effect) ---
# We'll plot RK4 as the primary "accurate" example
ax1.plot(t_points, z_rk4[:, 0], 'b-', label='Rabbits (RK4)', lw=2)
ax1.plot(t_points, z_rk4[:, 1], 'r-', label='Foxes (RK4)', lw=2)

# Add Euler just to show the drift/error
ax1.plot(t_points, z_euler[:, 0], 'b--', alpha=0.3, label='Rabbits (Euler Error)')
ax1.plot(t_points, z_euler[:, 1], 'r--', alpha=0.3, label='Foxes (Euler Error)')

ax1.set_title("Population Dynamics: The 'Lag' Effect")
ax1.set_xlabel("Time")
ax1.set_ylabel("Population Count")
ax1.legend(loc='upper right', fontsize='small')
ax1.grid(True, alpha=0.3)

# --- PLOT 2: Phase Portrait (The Stability Check) ---
# This plots Rabbits on X and Foxes on Y
ax2.plot(z_euler[:, 0], z_euler[:, 1], 'r--', label='Euler (Unstable Spiral)', alpha=0.5)
ax2.plot(z_rk4[:, 0], z_rk4[:, 1], 'b-', label='RK4 (Stable Cycle)', lw=2)
ax2.plot(z_scipy[:, 0], z_scipy[:, 1], 'g:', label='SciPy (Adaptive)', lw=2)

ax2.set_title("Phase Portrait: Stability Comparison")
ax2.set_xlabel("Rabbit Population (x)")
ax2.set_ylabel("Fox Population (y)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
