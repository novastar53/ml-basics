import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate correlated data (ellipse shape)
np.random.seed(0)
mean = [0, 0]
cov = [[3, 2], [2, 2]]  # covariance matrix with off-diagonal terms (correlated)
X = np.random.multivariate_normal(mean, cov, size=500)

# Plot original correlated data
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title('Original Correlated Data')
plt.axis('equal')

# Step 2: Compute covariance and eigendecomposition
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.cov(X_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Step 3: Rotate data using eigenvectors
X_rotated = X_centered @ eigenvectors

# Plot rotated data
plt.subplot(1, 3, 2)
plt.scatter(X_rotated[:, 0], X_rotated[:, 1], alpha=0.5, color='g')
plt.title('Rotated (Decorrelated) Data')
plt.axis('equal')

# Check covariance after rotation
cov_rotated = np.cov(X_rotated, rowvar=False)
print("Covariance after rotation:\n", cov_rotated.round(4))

# Step 4: Whitening (rotating and scaling)
X_whitened = X_rotated / np.sqrt(eigenvalues)

# Plot whitened data
plt.subplot(1, 3, 3)
plt.scatter(X_whitened[:, 0], X_whitened[:, 1], alpha=0.5, color='r')
plt.title('Whitened Data')
plt.axis('equal')

# Check covariance after whitening
cov_whitened = np.cov(X_whitened, rowvar=False)
print("\nCovariance after whitening:\n", cov_whitened.round(4))

plt.tight_layout()
plt.show()