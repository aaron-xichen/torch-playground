local optim = require 'optim'
local utee = require 'utee'

local M = {}
local Trainer = torch.class('quantization.Trainer', M)

function Trainer:getBestStat()
    keys = {'bestTop1Err', 'bestTop5Err'}
    vals = {math.huge, math.huge}
    return keys, vals
end

function Trainer:__init(model, criterion, optimState, opt, trainDataLoader, valDataLoader)
    self.model = model
    self.model:evaluate()
    self.orders = utee.topologicalOrder(self.model)
    self.opt = opt

    self.trainDataLoader = trainDataLoader
    self.valDataLoader = valDataLoader
end

function Trainer:quantizeParam()
    if self.opt.convNBits ~= -1 and self.opt.fcNBits ~= -1 then
        print('=> Quantizing weights and bias') 
        for i=1, #self.orders do
            local weight = self.orders[i].weight
            local bias = self.orders[i].bias
            local layerName = torch.typename(self.orders[i])
            if weight and bias then
                if layerName ~= 'nn.SpatialBatchNormalization' or self.opt.isQuantizeBN then
                    local paramNBits
                    if layerName == 'cudnn.SpatialConvolution' 
                        or layerName == 'nn.SpatialConvolution' 
                        or layerName == 'nn.SpatialConvolutionFixedPoint' 
                        or layerName == 'nn.SpatialBatchNormalization' then
                        paramNBits = self.opt.convNBits
                    elseif layerName == 'nn.Linear' then
                        paramNBits = self.opt.fcNBits
                    else
                        assert(nil, "Unknow layer type " .. layerName)
                    end

                    if torch.abs(weight):max() ~= 0  then
                        local weightShiftBits = -torch.ceil(torch.log(torch.abs(weight):max()) / torch.log(2))
                        weight:copy(2^-weightShiftBits * utee.quantization(weight * 2^weightShiftBits, 1, paramNBits-1))
                        self.orders[i].weightShiftBits = weightShiftBits
                    end

                    if torch.abs(bias):max() ~= 0 then
                        local biasShiftBits = -torch.ceil(torch.log(torch.abs(bias):max()) / torch.log(2))
                        bias:copy(2^-biasShiftBits * utee.quantization(bias * 2^biasShiftBits, 1, paramNBits-1))
                        self.orders[i].biasShiftBits = biasShiftBits
                    end

                    print(layerName, self.orders[i].weightShiftBits, self.orders[i].biasShiftBits)
                end
            end
        end
        print(("=> Quantization Info, ConvParam: %d bits, FcParam: %d bits, Activation: %d bits")
            :format(self.opt.convNBits, self.opt.fcNBits, self.opt.actNBits))
    end
end

function Trainer:castTensorType()
    for i=1, #self.orders do
        if self.opt.device == 'cpu' and self.opt.tensorType == 'double' then
            self.orders[i]:type('torch.DoubleTensor')
        end
        if self.orders[i].weight then
            print(torch.typename(self.orders[i].weight))
        end
    end
end

function Trainer:analyzeAct()
    if self.opt.actNBits ~= -1 then
        print("=> Analyzing activation distribution")
        for n, sample in self.valDataLoader:run() do
            self:copyInputs(sample)
            utee.analyzeAct(self.model, self.input, self.opt)
            if n >= self.opt.collectNSamples then
                self.valDataLoader:reset()
                break
            end
        end

        for i=1, #self.orders do
            if self.orders[i].actShiftBits then
                local layerName = torch.typename(self.orders[i])
                print(layerName, self.orders[i].actShiftBits)
            end
        end
        print('=> Analyzing done!')
    end
end

function Trainer:quantizationForward()
    if self.opt.device == 'cpu' and self.opt.tensorType == 'double' then
        utee.quantizationForward(self.model, self.input:double(), self.opt.actNBits)
    else 
        utee.quantizationForward(self.model, self.input, self.opt.actNBits)
    end
end

function Trainer:manualForward()
    if self.opt.device == 'cpu' and self.opt.tensorType == 'double' then
        self.model:forward(self.input:double())
    else 
        self.model:forward(self.input)
    end
end

function Trainer:val()
    local size = self.valDataLoader:size()
    local nCrops = self.opt.tenCrop and 10 or 1
    local top1Sum, top5Sum = 0.0, 0.0
    local N = 0

    self:quantizeParam()
    self:castTensorType()
    self:analyzeAct()

    -- forward
    for n, sample in self.valDataLoader:run() do
        self:copyInputs(sample)
        --torch.save('input.t7', sample.input:float())
        print('data: ', torch.sum(sample.input))
        if self.opt.actNBits == -1 then
            self:manualForward()
        else
            self:quantizationForward()
        end

        local top1, top5 = self:computeScore(self.model:get(#self.model).output, sample.target, nCrops)
        top1Sum = top1Sum + top1
        top5Sum = top5Sum + top5
        N = N + 1
        print((' | Val [%d/%d] Top1Err: %7.5f, Top5Err: %7.5f'):format(n, size, top1, top5))
    end

    print((' * Val Done, Top1Err: %7.3f  Top5Err: %7.3f'):format(top1Sum / N, top5Sum / N))
    print(("=> Quantization Info, ConvParam: %d bits, FcParam: %d bits, Activation: %d bits")
        :format(self.opt.convNBits, self.opt.fcNBits, self.opt.actNBits))
    vals = {top1Sum / N, top5Sum / N}
    return vals
end

------------ helper function ----------------
function Trainer:train(epoch)
    assert(nil, 'Not Implement')
end

function Trainer:computeScore(output, target, nCrops)
    if nCrops > 1 then
        output = output:view(output:size(1) / nCrops, nCrops, output:size(2)):sum(2):squeeze(2)
    end

    local batchSize = output:size(1)
    local _ , predictions = output:float():sort(2, true) -- descending
    local correct = predictions:eq(target:long():view(batchSize, 1):expandAs(output))
    local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)
    local len = math.min(5, correct:size(2))
    local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)
    return top1 * 100, top5 * 100
end

function Trainer:copyInputs(sample)
    if self.opt.device == 'gpu' then
        self.input = sample.input:cuda()
        self.target = sample.target:cuda()
    else
        self.input = sample.input
        self.target = sample.target
    end
end

return M.Trainer
