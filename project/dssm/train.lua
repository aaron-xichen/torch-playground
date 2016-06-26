local optim = require 'optim'
local M = {}
local Trainer = torch.class('dssm.Trainer', M)

function Trainer:cutoffFullToLow(nBits)
    local lowPrecisionModules = self.model:findModules('nn.Linear')
    assert(#self.fullPrecisionModules == #lowPrecisionModules, 'modules do not match')
    for k, v in pairs(lowPrecisionModules) do
        if nBits ~= -1 then
            v.weight:copy(self:quantization(self.fullPrecisionModules[k].weight, nBits))
        else
            v.weight:copy(self.fullPrecisionModules[k].weight)
        end
        v.bias:copy(self.fullPrecisionModules[k].bias) -- bias does not need quantization
    end
end

function Trainer:copyLowtoFull()
    local lowPrecisionModules = self.model:findModules('nn.Linear')
    assert(#self.fullPrecisionModules == #lowPrecisionModules, 'modules do not match')
    for k, v in pairs(self.fullPrecisionModules) do
        v.weight:copy(lowPrecisionModules[k].weight)
        v.bias:copy(lowPrecisionModules[k].bias)
    end
end

function Trainer:quantization(x, nBits)
    local M = 2 ^ nBits - 1
    local delta = 2 ^ -(nBits - 1)
    local sign = torch.sign(x)
    local floor = torch.floor(torch.abs(x) / delta + 0.5)
    local min = torch.cmin(floor, (M - 1) / 2.0)
    local raw = torch.mul(torch.cmul(min, sign), delta)
    return torch.clamp(raw, -1, 1)
end

function Trainer:getBestStat()
    keys = {'bestFloatLoss', 'bestQuantLoss'}
    vals = {math.huge, math.huge}
    return keys, vals
end

function Trainer:__init(model, criterion, optimState, opt)
    self.model = model
    self.criterion = criterion
    self.optimState = optimState or {
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        nesterov = opt.nesterov,
        dampening = 0.0,
        weightDecay = opt.weightDecay,
    }
    self.opt = opt
    self.params, self.gradParams = self.model:getParameters()

    self.fullPrecisionModules = {}
    for k, v in pairs(self.model:findModules('nn.Linear')) do
        self.fullPrecisionModules[k] = v:clone()
    end
end

function Trainer:train(epoch, dataloader)
    self.optimState.learningRate = self:learningRate(epoch)

    local function feval()
        return self.criterion.output, self.gradParams
    end

    local trainSize = dataloader:size()
    local floatLossSum, quantLossSum = 0.0, 0.0
    local N = 0

    print('=> Training epoch # ' .. epoch .. ' LR: ' .. self.optimState.learningRate)
    self.model:training()

    local timer = torch.Timer()
    local overall = torch.Timer()
    local wholeEpoch = torch.Timer()
    for sample in dataloader:nextBatch() do
        local dataTime = timer:time().real
        timer:reset()

        self:copyInputs(sample)
        local copyTime = timer:time().real
        timer:reset()

        -- perform float test
        self:cutoffFullToLow(-1)
        self.model:forward(self.input)
        local floatLoss = self.criterion:forward(self.model.output, self.target)
                
        -- cutoff 
        self:cutoffFullToLow(self.opt.nBits)
        local quantizationTime = timer:time().real
        timer:reset()

        -- forward
        self.model:forward(self.input)
        local quantLoss = self.criterion:forward(self.model.output, self.target)

        -- backward
        self.model:zeroGradParameters()
        self.model:backward(self.input, self.criterion:backward(self.model.output, self.target))

        -- update
        self:cutoffFullToLow(-1)
        optim.sgd(feval, self.params, self.optimState)
        self:copyLowtoFull()
        
        local trainTime = timer:time().real
        timer:reset()
        
        floatLossSum = floatLossSum + floatLoss
        quantLossSum = quantLossSum + quantLoss
        N = N + 1
        print((' | Train [%d/%d] Data:%.3f, CopyToGPU:%.3f, Quant:%.3f, FP-BP:%.3f, Total: %.3f, FloatLoss: %.3f, QuantLoss: %.3f')
            :format(N, trainSize, dataTime, copyTime, quantizationTime, trainTime, overall:time().real, floatLoss, quantLoss))

        assert(self.params:storage() == self.model:parameters()[1]:storage())

        overall:reset()
        timer:reset()
    end
    print((' | Train Done, Float Loss: %.3f, QuantLoss: %.3f, cost: %.3f')
        :format(floatLossSum / N, quantLossSum / N, wholeEpoch:time().real))
end

function Trainer:val(dataloader)
    local floatLossSum, quantLossSum = 0.0, 0.0
    local N = 0
    self.model:evaluate()
    local wholeEpoch = torch.Timer()
    for sample in dataloader:nextBatch() do
        self:copyInputs(sample)

        self:cutoffFullToLow(-1)
        local floatLoss = self.criterion:forward(self.model:forward(self.input), self.target)
        floatLossSum = floatLossSum + floatLoss

        self:cutoffFullToLow(self.opt.nBits)
        local quantLoss = self.criterion:forward(self.model:forward(self.input), self.target)
        quantLossSum = quantLossSum + quantLoss

        N = N + 1
    end
    vals = {floatLossSum / N, quantLossSum / N}
    print((' * Val Done, Float Loss: %.3f, QuantLoss: %.3f, cost: %.3f'):format(vals[1], vals[2], wholeEpoch:time().real))
    return vals
end

function Trainer:copyInputs(sample)
    timer = torch.Timer()
    self.input = self.input or (self.opt.nGPU == 1
        and torch.CudaTensor()
        or cutorch.createCudaHostTensor())
    self.target = self.target or torch.CudaTensor()
    self.input:resize(sample.input:size()):copy(sample.input)
    self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
    --local decay = epoch >= 5 and 2 or epoch >= 3 and 1 or 0
    --return self.opt.LR * math.pow(0.1, decay)
    return self.opt.LR
end

return M.Trainer
