local CicularShift, parent = torch.class('nn.CicularShift', 'nn.Module')

function CicularShift:__init(dim, offset, cycleSize)
    parent.__init(self)
    
    assert(dim, 'dim is nil value')
    assert(offset, 'offset is nil value')
    assert(cycleSize, 'cycleSize is nil value')
    self.dim = dim
    local order = torch.range(0, cycleSize - 1):long()
    self.order = torch.mod(order + offset, cycleSize) + 1
    _, self.inv_order = self.order:sort()
    self.offset = offset
end

function CicularShift:updateOutput(input)
    assert(self.dim <= input:dim(), 'Dimension is out of bound')
    self.output = input:index(self.dim, self.order):contiguous()
    return self.output
end

function CicularShift:updateGradInput(input, gradOutput)
    assert(self.dim <= input:nDimension(), 'Dimension is out of bound')
    self.gradInput = gradOutput:index(self.dim, self.inv_order):contiguous()
    return self.gradInput
end

function CicularShift:type(t)
    if t == 'torch.CudaTensor' then
        self.order = self.order:cuda()
        self.inv_order = self.inv_order:cuda()
    else
        self.order = self.order:long()
        self.inv_order = self.inv_order:long()
    end
    return self
end