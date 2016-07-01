local optim = require 'optim'
local utee = require 'utee'

local M = {}
local Trainer = torch.class('template.Trainer', M)

function Trainer:getBestStat()
    keys = {'bestTop1Err', 'bestTop5Err'}
    vals = {math.huge, math.huge}
    return keys, vals
end

function Trainer:__init(model, criterion, optimState, opt, trainDataLoader, valDataLoader)
    self.model = model
    self.shadow = self.model:clone()

    self.criterion = criterion
    self.optimState = optimState or {
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        nesterov = true,
        dampening = 0.0,
        weightDecay = opt.weightDecay,
    }
    self.opt = opt
    self.params, self.gradParams = self.shadow:getParameters()
    self.trainDataLoader = trainDataLoader
    self.valDataLoader = valDataLoader

    self.relatedLayers = {}
    self.relatedLayers['cudnn.SpatialConvolution'] = true
    self.relatedLayers['nn.SpatialConvolution'] = true
    --self.relatedLayers['cudnn.SpatialCrossMapLRN'] = true
    --self.relatedLayers['nn.SpatialCrossMapLRN'] = true
    self.relatedLayers['nn.Linear'] = true
    --self.relatedLayers['nn.ReLU'] = true
    --self.relatedLayers['cudnn.ReLU'] = true
    self.relatedLayers['cudnn.SoftMax'] = true
    self.relatedLayers['nn.SoftMax'] = true
    self.collectNSamples = 10
end

function Trainer:paramQuantization()
    if self.opt.convNBits ~= -1 and self.opt.fcNBits ~= -1 then
        -- fix point weights and bias
        print('=> Quantizing weights and bias') 
        self.paramShiftNBits = {}
        for i=1, #self.model do
            local weight = self.model:get(i).weight
            local bias = self.model:get(i).bias
            if weight and bias then
                local layerName = torch.typename(self.model:get(i))
                local paramNBits
                if layerName == 'cudnn.SpatialConvolution' or layerName == 'nn.SpatialConvolution' then
                    paramNBits = self.opt.convNBits
                elseif layerName == 'nn.Linear' then
                    paramNBits = self.opt.fcNBits
                else
                    assert(nil, "Unknow layer type " .. layerName)
                end

                self.paramShiftNBits[i] = {}
                self.paramShiftNBits[i][1] = torch.ceil(torch.log(torch.abs(weight):max()) / torch.log(2))
                self.paramShiftNBits[i][2] = torch.ceil(torch.log(torch.abs(bias):max()) / torch.log(2))

                print(weight:min(), weight:max(), self.paramShiftNBits[i][1])
                print(bias:min(), bias:max(), self.paramShiftNBits[i][2])

                local weightShiftTo = weight * 2 ^ -self.paramShiftNBits[i][1]
                local weightQuantization = utee.quantization(weightShiftTo, 1, paramNBits - 1)
                local weightShiftBack = weightQuantization * 2 ^ self.paramShiftNBits[i][1]
                print('sample: ', weight:view(-1)[1], weightShiftBack:view(-1)[1])
                self.shadow:get(i).weight:copy(weightShiftBack)

                local biasShiftTo = bias * 2 ^ -self.paramShiftNBits[i][2]
                local biasQuantization = utee.quantization(biasShiftTo, 1, paramNBits - 1)
                local biasShiftBack = biasQuantization * 2 ^ self.paramShiftNBits[i][2]
                print('sample: ', bias:view(-1)[1], biasShiftBack:view(-1)[1])
                self.shadow:get(i).bias:copy(biasShiftBack)
            end
        end
        print(("=> Quantization Info, ConvParam: %d bits, FcParam: %d bits, Activation: %d bits")
            :format(self.opt.convNBits, self.opt.fcNBits, self.opt.actNBits))
    end
end

function Trainer:actAnalysis()
    if self.opt.actNBits ~= -1 then
        print("=> Analyzing activation distribution")
        local cache = {}

        print(('=> Sampling %d data points'):format(self.collectNSamples))
        for n, sample in self.valDataLoader:run() do
            self:copyInputs(sample)
            self.shadow:forward(self.input)
            for i=1, #self.shadow do
                local layerName = torch.typename(self.shadow:get(i))
                if self.relatedLayers[layerName] then
                    if not cache[i] then
                        cache[i] = {}
                    end
                    table.insert(cache[i], self.shadow:get(i).output)
                end
            end
            if n >= self.collectNSamples then
                self.valDataLoader:reset()
                break
            end
        end

        print('=> Computing number of shifting bits')
        self.actShiftNBits = {}
        for k, v in pairs(cache) do
            self.actShiftNBits[k] = utee.maxShiftNBitsTable(v)
        end
        print('=> Analyzing done!')
    end
end

function Trainer:quantizationForward()
    for i=1, #self.shadow do
        if i == 1 then
            self.shadow:get(i):forward(self.input)
        else
            self.shadow:get(i):forward(self.shadow:get(i-1).output)
        end
        if self.actShiftNBits[i] then
            local shiftToVal = 2^-self.actShiftNBits[i] * self.shadow:get(i).output
            local quantizationVal = utee.quantization(
                shiftToVal, 
                1,
                self.opt.actNBits - 1
            )
            local shiftBackVal = 2^self.actShiftNBits[i] * quantizationVal
            self.shadow:get(i).output:copy(shiftBackVal)
        end
    end
end

function Trainer:manualForward()
    for i=1, #self.shadow do
        if i == 1 then
            self.shadow:get(i):forward(self.input)
        else
            self.shadow:get(i):forward(self.shadow:get(i-1).output)
        end
        local layerName = torch.typename(self.shadow:get(i))
        if self.relatedLayers[layerName] then
            local meanVal = torch.mean(self.shadow:get(i).output)
            local minVal = self.shadow:get(i).output:min()
            local maxVal = self.shadow:get(i).output:max()
            --print(layerName, meanVal, minVal, maxVal)
        end
    end
end

function Trainer:train(epoch)
    local size = self.trainDataLoader:size()
    self.optimState.learningRate = self:learningRate(epoch)

    local function feval()
        return self.criterion.output, self.gradParams
    end
    print('=> Training epoch # ' .. epoch .. ' LR: ' .. self.optimState.learningRate)
    local lossSum = 0.0
    local N = 0

    self.shadow:training()
    self.trainDataLoader:reset()
    for n, sample in self.trainDataLoader:run() do
        self:copyInputs(sample)

        self:paramQuantization()
        self.shadow:forward(self.input)

        local loss = self.criterion:forward(self.shadow.output, self.target)
        self.shadow:zeroGradParameters()
        self.criterion:backward(self.shadow.output, self.target)
        self.shadow:backward(self.input, self.criterion.gradInput)

        utee.copyTo(self.model, self.shadow)
        optim.sgd(feval, self.params, self.optimState)
        utee.copyTo(self.shadow, self.model)

        print((' | Train [%d/%d] Loss: %3.3f'):format(n, size, loss))
        lossSum = lossSum + loss
        N = N + 1
        if N >= 20 then break end
    end
    print((' | Train Done, Loss: %3.3f'):format(lossSum / N))
    assert(self.params:storage() == self.shadow:parameters()[1]:storage())
end

function Trainer:val()
    local size = self.valDataLoader:size()
    local nCrops = self.opt.tenCrop and 10 or 1
    local top1Sum, top5Sum = 0.0, 0.0
    local N = 0

    self.shadow:evaluate()
    self.valDataLoader:reset()

    -- init
    self:paramQuantization()
    if not self.actShiftNBits then
        self:actAnalysis()
    end

    -- forward
    for n, sample in self.valDataLoader:run() do
        self:copyInputs(sample)
        if self.opt.actNBits == -1 then
            self:manualForward()
        else
            self:quantizationForward()
        end

        local top1, top5 = self:computeScore(self.shadow:get(#self.shadow).output, sample.target, nCrops)
        top1Sum = top1Sum + top1
        top5Sum = top5Sum + top5
        N = N + 1
        print((' | Val [%d/%d] Top1Err: %7.5f, Top5Err: %7.5f'):format(n, size, top1, top5))
        --if N >= 1 then break end
    end

    print((' * Val Done, Top1Err: %7.3f  Top5Err: %7.3f'):format(top1Sum / N, top5Sum / N))
    print(("=> Quantization Info, ConvParam: %d bits, FcParam: %d bits, Activation: %d bits")
        :format(self.opt.convNBits, self.opt.fcNBits, self.opt.actNBits))
    vals = {top1Sum / N, top5Sum / N}
    return vals
end

function Trainer:computeScore(output, target, nCrops)
    if nCrops > 1 then
        -- Sum over crops
        output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
        --:exp()
        :sum(2):squeeze(2)
    end

    -- Coputes the top1 and top5 error rate
    local batchSize = output:size(1)

    local _ , predictions = output:float():sort(2, true) -- descending

    -- Find which predictions match the target
    local correct = predictions:eq(
        target:long():view(batchSize, 1):expandAs(output))

    -- Top-1 score
    local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

    -- Top-5 score, if there are at least 5 classes
    local len = math.min(5, correct:size(2))
    local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

    return top1 * 100, top5 * 100
end

function Trainer:copyInputs(sample)
    if self.opt.device == 'gpu' then
        self.input = self.input or (self.opt.nGPU == 1
            and torch.CudaTensor()
            or cutorch.createCudaHostTensor())
        self.target = self.target or torch.CudaTensor()

        self.input:resize(sample.input:size()):copy(sample.input)
        self.target:resize(sample.target:size()):copy(sample.target)
    else
        self.input = sample.input
        self.target = sample.target
    end
end

function Trainer:learningRate(epoch)
    decay = epoch >= 375 and 4 or epoch >= 350 and 3 or epoch >= 300 and 2 or epoch >=200 and 1 or 0
    return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
