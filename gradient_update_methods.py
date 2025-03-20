import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

lr = 0.001

def f(X, Y):
    return 50*(X**2) + 0.01*(Y**2)

def g_f(X, Y):
    return np.array([100*X, 0.02*Y])

def h_inv():
    H = np.array([[100, 0], [0, 0.02]])
    H_inv = np.linalg.inv(H)
    return H_inv

def grad_step(x, y,  lr=lr):
    g = g_f(x, y)
    return [x - lr*g[0],  y - lr*g[1]]

def newton_step(x, y, lr=lr):
    g = g_f(x, y)
    h_i = h_inv()
    g_prime = h_i @ g
    return [ x - lr*g_prime[0], y - lr*g_prime[1]]

acc_g = None
def momentum_step(x, y, lr=lr, m=0.99):
    global acc_g
    if acc_g is None:
        return grad_step(x, y)
    g_prime = m*acc_g + g_f(x, y)
    acc_g = g_prime
    return [ x - lr*g_prime[0], y - lr*g_prime[1] ]





X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

num_frames = 100

X_point = []
Y_point = []
x = -5
y = 0
for i in range(num_frames):
    x_next, y_next = momentum_step(x, y)
    X_point.append(x_next)
    Y_point.append(y_next)
    x = x_next
    y = y_next


Z_point = list(map(f, X_point, Y_point))

marker = np.array(list(zip(list(range(num_frames)), X_point, Y_point, Z_point)))

# Create the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set axis limits
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_zlim([-6, 1000])

ax.plot_surface(X, Y, Z, cmap='viridis', zorder=1000, alpha=0.5)

# Create the point that will be animated
point, = ax.plot([], [], [], 'ro', markersize=8, color='red', markerfacecolor='red', zorder=0)

# Initialization function
def init():
    point.set_data([], [])
    point.set_3d_properties([])
    return point,

# Animation function
def update(frame):
    _, x, y, z = marker[frame]
    point.set_data([x], [y])
    point.set_3d_properties([z]) 
    ax.set_title(f"Step: {frame}")
    ax.figure.canvas.draw()  # Force redraw
    return point,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=500)

# Display animation
ani.save('gradient_descent.gif', writer='ffmpeg', fps=30)

plt.show()