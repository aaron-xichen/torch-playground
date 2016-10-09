require 'nn';

m1 = nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1)
m1.weight:uniform(-1, 1)
m1.bias:uniform(-1, 1)
m2 = nn.SpatialConvolutionFixedPoint(3, 16, 3, 3, 1, 1, 1, 1)
m2.weight:copy(m1.weight)
m2.bias:copy(m1.bias)


input = torch.Tensor(64, 3, 32, 32):uniform(-1, 1)

y1 = m1:forward(input)
y2 = m2:forward(input)

print(torch.sum(y1 - y2))
