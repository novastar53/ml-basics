import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate data
X = np.linspace(-5, 5, 10)
Y = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(X, Y)

# Generate function
def f(X, Y):
    return 50*(X**2) + 10*(Y**2)

# Generate gradient
def g_x(X):
    return 100*X/1000

def g_y(Y):
    return 20*Y/1000

# Generate Hessian
def h_xx():
    return 100

def h_xy():
    return 0

def h_yx():
    return 0

def h_yy():
    return 20

Z = f(X, Y)
G_x = g_x(X) 
G_y = g_y(Y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Surface Plot')

ax.quiver(X, Y, 0, G_x, G_y, 0, color='r', normalize=False)
plt.show()