require 'nn'
m = nn.SpatialConvolutionFixedPoint(3, 3, 3, 3, 1, 1, 1, 1)
m:type('torch.IntTensor')
m.weight:random(-10, 10)
m.bias:random(-10, 10)
print(m.weight)
print(m.bias)
x = torch.IntTensor(8, 3, 20, 20):random(-10, 10)

print(m:forward(x))