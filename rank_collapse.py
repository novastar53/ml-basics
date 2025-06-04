import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Parameters
matrix_size = 50  # Dimension of square matrices
num_matrices = 200  # Number of matrices to multiply

# Initialize a random vector
x = np.random.randn(matrix_size, 1)
x /= np.linalg.norm(x)

# Generate random matrices and multiply
singular_values_over_time = []

# Start with identity
product_matrix = np.eye(matrix_size)

for _ in range(num_matrices):
    random_matrix = np.random.randn(matrix_size, matrix_size) / np.sqrt(matrix_size)
    product_matrix = random_matrix @ product_matrix
    
    # Record singular values
    singular_values = np.linalg.svd(product_matrix, compute_uv=False)
    singular_values_over_time.append(singular_values)

# Plot singular values
plt.figure(figsize=(10, 6))

for i in range(5):  # plot top 10 singular values
    plt.plot([sv[i] for sv in singular_values_over_time], label=f'Singular value {i+1}')

plt.xlabel('Number of matrix multiplications')
plt.ylabel('Singular values (log scale)')
plt.title('Rank Collapse in Product of Random Matrices')
plt.legend()
plt.grid(True)
plt.show()