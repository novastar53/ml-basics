import torch

# input vector
x = torch.tensor(3)

# weight
w = torch.tensor(2.0, requires_grad=True)

# bias
b = torch.tensor(5.0, requires_grad=True)

# function
y = w*x+b

# function output
print(y)

# compute gradients
y.backward()

# print the gradients of y w.r.t each parameter
dw = w.grad
db = b.grad
print("dw", w.grad)
print("db", b.grad)


