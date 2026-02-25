import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Define the Physics (The Derivative)
def model(t, y, k=0.6):
    return -k * y

# 2. Parameters
y0 = 100        # Starting amount
t0, tf = 0, 10  # Start and end time
n_steps = 15    # Low step count to show how methods diverge
t_points = np.linspace(t0, tf, n_steps)
h = t_points[1] - t_points[0] # Step size

# --- METHOD A: Euler's Method ---
y_euler = np.zeros(n_steps)
y_euler[0] = y0
for i in range(n_steps - 1):
    # y_next = y_now + slope * dt
    y_euler[i+1] = y_euler[i] + model(t_points[i], y_euler[i]) * h

# --- METHOD B: Runge-Kutta (RK4) ---
y_rk4 = np.zeros(n_steps)
y_rk4[0] = y0
for i in range(n_steps - 1):
    ti = t_points[i]
    yi = y_rk4[i]
    
    k1 = model(ti, yi)
    k2 = model(ti + h/2, yi + h*k1/2)
    k3 = model(ti + h/2, yi + h*k2/2)
    k4 = model(ti + h, yi + h*k3)
    
    y_rk4[i+1] = yi + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# --- METHOD C: SciPy's solve_ivp (The Pro Tool) ---
# RK45 is the default (an adaptive version of Runge-Kutta)
sol = solve_ivp(model, [t0, tf], [y0], t_eval=t_points)
y_scipy = sol.y[0]

# --- THE TRUTH: Exact Analytical Solution ---
y_exact = y0 * np.exp(-0.6 * t_points)

# 3. Visualization
plt.figure(figsize=(10, 6))
plt.plot(t_points, y_exact, 'k', label='Exact Solution (Truth)', lw=3, alpha=0.3)
plt.plot(t_points, y_euler, 'ro--', label="Euler (The 'Rough' Guess)")
plt.plot(t_points, y_rk4, 'bs-', label="RK4 (The Workhorse)")
plt.plot(t_points, y_scipy, 'g^', label="SciPy (Adaptive RK45)")
plt.title("ODE Solver Comparison: Radioactive Decay")
plt.xlabel("Time")
plt.ylabel("Amount Remaining")
plt.legend()
plt.grid(True)
plt.show()
