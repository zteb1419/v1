import torch

# Direct creation from data
data = [[1, 2, 3, 4], [3, 4, 5, 6]]
x_data = torch.tensor(data)
print(x_data)
# Creation from a desired shape
shape = (2, 3)
ones = torch.ones(shape)
print(ones)
zeros = torch.zeros(shape)
print(zeros)
random = torch.randn(shape)
print(random)
# Creation by mimicking another tensor
template = torch.tensor([[1, 2], [3, 4]])
rand_like = torch.randn_like(template, dtype=torch.float)
print(template, "\n", rand_like)
# Three attributes: shape, type, device
tensor = torch.randn(2, 3)
print(f"Shape: {tensor.shape}")
print(f"Datatype:  {tensor.dtype}")
print(f"Device: {tensor.device}")
# ***Data vs parameter ***
# Standard data tensor
x_data = torch.tensor([[1., 2.], [3., 4.]])
# A parameter tensor (gradients required)
w = torch.tensor([[1.0], [2.0]], requires_grad=True)
print(f"Data tensor requires_grad: {x_data.requires_grad}")
print(f"Parameter tensor requires_grad: {w.requires_grad}")
# Building the graph z=x*y, where y=a+b
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(4.0, requires_grad=True)
y = a + b
z = x * y
print(f"Result z: {z}")
print(f"grad_fn for z: {z.grad_fn}")
print(f"grad_fn for y: {y.grad_fn}")
print(f"grad_fn for a: {a.grad_fn}")
# * multiplication vs @ "matrix" multiplication 
m1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
m2 = torch.tensor([[7, 8], [9, 10], [11, 12]])
matrix_product = m1 @ m2
print(matrix_product)
# Reduction operations
scores = torch.tensor([[10., 20., 30.], [5., 10., 15.]])
average_score = scores.mean()
print(f"Overall Mean: {average_score}")
avg_per_assignment = scores.mean(dim=0)
print(f"Average per assignment: {avg_per_assignment}")
avg_per_student = scores.mean(dim=1)
print(f"Average per student: {avg_per_student}")
print(f"Shape of scores tensor: {scores.shape}, shape of avg_per_student tensor: {avg_per_student.shape}")
