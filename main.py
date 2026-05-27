from sonnet_core.tensor import Tensor

x = Tensor(2)
y = Tensor(3)

z = x * y

print(z)

print(z._prev)
print(z._op)