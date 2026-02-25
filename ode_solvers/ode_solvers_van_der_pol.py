import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Define the Van der Pol System
# mu = 1000 makes it "Stiff"
def vanderpol(t, z, mu=1000):
    x, y = z
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

# 2. Parameters
z0 = [2, 0]        # Initial conditions
t_span = [0, 3000] # Long time to see the slow-fast behavior

# --- Attempt 1: Standard RK45 (The default we used before) ---
# This will take a LONG time or fail because it's not meant for stiffness
sol_rk45 = solve_ivp(vanderpol, t_span, z0, method='RK45')

# --- Attempt 2: BDF (Backward Differentiation Formula) ---
# This is a "Stiff" solver. It uses implicit math to stay stable.
sol_stiff = solve_ivp(vanderpol, t_span, z0, method='BDF')

# 3. Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plotting the "Snap" behavior (Position x over time)
ax1.plot(sol_stiff.t, sol_stiff.y[0], 'g-', label='Stiff Solver (BDF)')
ax1.set_title(f"Van der Pol Oscillator (mu=1000): The 'Slow-Fast' Dynamics")
ax1.set_ylabel("Position (x)")
ax1.legend()

# Comparing Step Sizes (The "Aha!" Moment)
ax2.plot(sol_rk45.t[:-1], np.diff(sol_rk45.t), 'ro', markersize=2, label='RK45 Step Sizes', alpha=0.5)
ax2.plot(sol_stiff.t[:-1], np.diff(sol_stiff.t), 'go', markersize=2, label='BDF Step Sizes', alpha=0.5)
ax2.set_yscale('log')
ax2.set_ylabel("Step Size (dt)")
ax2.set_xlabel("Time")
ax2.set_title("Solver Efficiency: How hard is the solver working?")
ax2.legend()


# --- PLOT 3: Phase Portrait (x vs y) ---
plt.figure(figsize=(8, 8))

# Plotting the BDF (Stiff) results
plt.plot(sol_stiff.y[0], sol_stiff.y[1], 'g-', lw=2, label='BDF Phase Path')

# Adding arrows to show the direction of "flow"
# We'll pick a few indices to place arrows
for i in [10, 50, 100, 150]:
    plt.arrow(sol_stiff.y[0, i], sol_stiff.y[1, i],
              sol_stiff.y[0, i+1]-sol_stiff.y[0, i],
              sol_stiff.y[1, i+1]-sol_stiff.y[1, i],
              shape='full', lw=0, length_includes_head=True, head_width=0.1, color='black')

plt.title(f"Stiff Phase Portrait (mu={1000})")
plt.xlabel("Position (x)")
plt.ylabel("Velocity (y)")
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.legend()
plt.show()

