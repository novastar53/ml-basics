import torch
import torch.nn.functional as F

# The input tensor of size b,t,k
# b = batch size
# t = sequence length
# k = embedding size
x = torch.randn([3,10,5])

# Multiply the input tensor with a transposed
# version of itself to get the unnormalized 
# self-attention weights
raw_weights = torch.bmm(x, x.transpose(1,2))

# Do a row-wise softmax to obtain the 
# the normalized weights
weights = F.softmax(raw_weights, dim=2)

# Multiply the weights by the input
# to obtain the output sequence.
# Thus, the output sequence is just a
# weighted sum of the input sequence
y = torch.bmm(weights, x)

# print the results
# As you can see, the output is of the same
# dimension as the input, but the values 
# have been transformed by interactions 
# between each pair of words
print("inputs", x)
print("outputs", y)

