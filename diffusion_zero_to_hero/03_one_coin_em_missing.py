"""
Section 3: One Coin EM with Missing Data

This demonstrates the Expectation-Maximization algorithm when:
- The coin bias theta is UNKNOWN
- Some flip outcomes are MISSING (hidden/latent)

The chicken-and-egg problem:
- To estimate theta, we need total heads (but missing flips are unknown)
- To infer missing flips, we need theta (but that's what we're estimating)

EM alternates between:
- E-step: Compute expected missing heads given current theta
- M-step: Update theta using expected total heads
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# True coin bias (unknown in practice)
true_theta = 0.6

# Total flips and missing flips
n_total = 100  # Total flips performed
n_missing = 40  # Number of flips with unknown outcomes
n_observed = n_total - n_missing  # Flips we actually see

# Simulate all flips (in reality, we wouldn't see the missing ones)
all_flips = np.random.binomial(1, true_theta, n_total)
observed_flips = all_flips[:n_observed]
missing_flips = all_flips[n_observed:]  # These are "hidden" / latent

# Count heads in observed flips
h_observed = np.sum(observed_flips)
h_missing_true = np.sum(missing_flips)  # For comparison only

print("=" * 60)
print("SETUP:")
print(f"True theta: {true_theta}")
print(f"Total flips: {n_total}")
print(f"Observed flips: {n_observed} (heads: {h_observed})")
print(f"Missing flips: {n_missing} (true heads: {h_missing_true})")
print(f"True total heads: {h_observed + h_missing_true}")
print("=" * 60)

# If we naively ignored missing data:
theta_naive = h_observed / n_observed
print(f"\nNaive estimate (ignoring missing): {theta_naive:.4f}")
print(f"True theta: {true_theta:.4f}")

# EM Algorithm
print("\n" + "=" * 60)
print("EM ALGORITHM")
print("=" * 60)

# Initialize theta randomly
theta = 0.5  # Our initial guess
max_iterations = 50
tolerance = 1e-6

theta_history = [theta]

for iteration in range(max_iterations):
    theta_old = theta

    # E-step: Compute expected heads in missing flips
    # If current theta is our best guess for P(heads),
    # then expected heads in missing flips = n_missing * theta
    expected_missing_heads = n_missing * theta

    # M-step: Update theta using expected total heads
    # MLE with "filled in" data: (observed + expected) / total
    expected_total_heads = h_observed + expected_missing_heads
    theta = expected_total_heads / n_total

    theta_history.append(theta)

    # Check convergence
    delta = abs(theta - theta_old)
    if iteration < 10 or delta > tolerance * 10:
        print(f"Iter {iteration:2d}: E[heads missing] = {expected_missing_heads:.2f}, "
              f"theta = {theta:.4f}, delta = {delta:.6f}")

    if delta < tolerance:
        print(f"\nConverged after {iteration + 1} iterations!")
        break

# Oracle estimate: what we'd get if we knew the missing data
theta_oracle = (h_observed + h_missing_true) / n_total

print("\n" + "=" * 60)
print("RESULTS:")
print(f"Initial theta guess:  0.5000")
print(f"EM final estimate:    {theta:.4f}")
print(f"Naive estimate:       {theta_naive:.4f}")
print(f"Oracle (know all):    {theta_oracle:.4f}")
print(f"True theta:           {true_theta:.4f}")
print(f"\nEM error:    {abs(theta - true_theta):.4f}")
print(f"Naive error: {abs(theta_naive - true_theta):.4f}")
print(f"Oracle error:{abs(theta_oracle - true_theta):.4f}")
print("=" * 60)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Convergence of theta
ax1 = axes[0]
ax1.plot(theta_history, 'o-', linewidth=2, markersize=6, label='EM estimate')
ax1.axhline(y=true_theta, color='r', linestyle='--', linewidth=2, label=f'True theta = {true_theta}')
ax1.axhline(y=theta_naive, color='orange', linestyle='--', linewidth=2, label=f'Naive = {theta_naive:.3f}')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Theta estimate', fontsize=12)
ax1.set_title('EM Convergence: Coin Bias with Missing Data', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Plot 2: Expected missing heads over iterations
expected_missing_history = [n_missing * th for th in theta_history]
ax2 = axes[1]
ax2.plot(expected_missing_history, 's-', linewidth=2, markersize=6, color='green', label='E[heads in missing]')
ax2.axhline(y=h_missing_true, color='r', linestyle='--', linewidth=2, label=f'True missing heads = {h_missing_true}')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Expected heads in missing flips', fontsize=12)
ax2.set_title('E-step: Expected Missing Heads', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_one_coin_em_missing.png', dpi=150)
print("\nSaved plot to: 03_one_coin_em_missing.png")

print("\n" + "=" * 60)
print("WHY EM CONVERGES TO 0.65 (NOT 0.6):")
print("")
print("EM finds the MAXIMUM LIKELIHOOD ESTIMATE given OBSERVED data.")
print("")
print("The observed data (60 flips, 39 heads) suggests theta = 0.65.")
print("EM correctly identifies this as the most likely value given")
print("what we actually observed.")
print("")
print("The difference from true theta (0.6) is due to SAMPLING VARIANCE:")
print("- True theta: 0.60 (the underlying coin bias)")
print("- Observed: 39/60 = 0.65 (random variation in the 60 observed flips)")
print("- Oracle: 63/100 = 0.63 (closer to truth with more data)")
print("")
print("WHY EM ISN'T USEFUL FOR THIS PROBLEM:")
print("")
print("For this simple one-coin problem, EM and the naive estimate")
print("converge to the SAME value. Why? Because the math works out that way:")
print("")
print("  E-step: Expected missing heads = k * theta")
print("  M-step: theta_new = (h_obs + k*theta) / n")
print("")
print("At convergence: theta = (h_obs + k*theta) / n")
print("               theta = h_obs / (n - k) = h_obs / n_obs")
print("")
print("This is exactly the NAIVE estimate! The iterations converge to")
print("the same closed-form solution we could have computed directly.")
print("")
print("EM's POWER comes from complex problems where the M-step does NOT")
print("have a closed-form solution. In the two-coin problem, the M-step")
print("requires solving for theta_1, theta_2, AND pi simultaneously —")
print("the iterative structure becomes essential, not redundant.")
print("")
print("KEY INSIGHT:")
print("EM handles missing data by alternating between:")
print("  E-step: Fill in missing data using current parameter estimate")
print("  M-step: Update parameters using 'completed' data")
print("")
print("Each iteration improves the estimate. The E-step computes")
print("the EXPECTATION of missing data; the M-step does MAXIMUM")
print("likelihood estimation with that expectation.")
print("=" * 60)
