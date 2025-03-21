import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

lr = 0.04

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

adagrad_eps = 1e-5
adagrad_s = np.array([adagrad_eps, adagrad_eps])
def adagrad_step(x, y, lr=lr):
    global adagrad_s
    g = g_f(x, y)
    adagrad_s = adagrad_s + g**2
    s = np.sqrt(1/adagrad_s)
    return [ x - lr*s[0]*g[0], y - lr*s[1]*g[1]]

rmsprop_eps = 1e-5
rmsprop_s = np.array([rmsprop_eps, rmsprop_eps])
def rmsprop_step(x, y, lr=lr, b=0.9):
    global rmsprop_s 
    g = g_f(x, y)
    rmsprop_s = b*rmsprop_s + (1 - b)*(g**2)
    s = rmsprop_s**(-0.5)
    return [ x - lr*s[0]*g[0], y - lr*s[1]*g[1]]


adam_eps = 1e-5
adam_m1 = np.array([0, 0])
adam_m2 = np.array([0, 0])
def adam_step(x, y, lr=lr, b1=0.9, b2=0.999):
    global adam_m1, adam_m2, adam_eps
    g = g_f(x, y)
    adam_m1 = b1*adam_m1 + (1 - b1)*g
    #adam_m1 = adam_m1/(1 - b1) # TODO: This scaling term shrinks the updates too much. Need to investigate.

    adam_m2 = b2*adam_m2 + (1 - b2)*(g**2)
    #adam_m2 = adam_m2/(1 - b2)
    s2 = 1 / (adam_eps + adam_m2**0.5)
    print(s2)

    return [x - lr*s2[0]*adam_m1[0], y - lr*s2[1]*adam_m1[1] ]


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
    x_next, y_next = adam_step(x, y)
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