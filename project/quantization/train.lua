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

                local weightNInt, biasNInt = paramNBits, paramNBits
                local weightOfr, biasOfr
                local weightShiftBits = torch.ceil(torch.log(torch.abs(weight):max()) / torch.log(2))
                local biasShiftBits = torch.ceil(torch.log(torch.abs(bias):max()) / torch.log(2))
                print(weight:min(), weight:max(), weightShiftBits)
                print(bias:min(), bias:max(), biasShiftBits)

                local weightShiftLeft = weight * 2 ^ -weightShiftBits
                for j=1,paramNBits do
                    weightOfr = utee.overflowRate(weightShiftLeft, j, paramNBits - j)
                    if weightOfr <= self.opt.overFlowRate then 
                        weightNInt = j
                        break 
                    end
                end
                local weightQuantization = utee.quantization(weightShiftLeft, weightNInt, paramNBits - weightNInt)
                local weightShiftRight = weightQuantization * 2 ^ weightShiftBits
                print('sample: ', weight:view(-1)[1], weightShiftRight:view(-1)[1])
                self.shadow:get(i).weight:copy(weightShiftRight)


                local biasShiftLeft = bias * 2 ^ -biasShiftBits
                for j=1,paramNBits do
                    biasOfr = utee.overflowRate(biasShiftLeft, j, paramNBits - j)
                    if biasOfr <= self.opt.overFlowRate then 
                        biasNInt = j
                        break 
                    end
                end
                local biasQuantization = utee.quantization(biasShiftLeft, biasNInt, paramNBits - biasNInt)
                local biasShiftRight = biasQuantization * 2 ^ biasShiftBits
                print('sample: ', bias:view(-1)[1], biasShiftRight:view(-1)[1])
                self.shadow:get(i).bias:copy(biasShiftRight)

                print(("%s, weight: %d.%d, bias: %d.%d")
                    :format(layerName, weightNInt, paramNBits - weightNInt, biasNInt, paramNBits - biasNInt))
            end
        end
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

        -- allocation
        print('=> Allocating bit length')
        self.allocationTable = {}
        self.shiftNBits = {}
        for k, v in pairs(cache) do
            self.shiftNBits[k] = utee.maxShiftNBitsTable(v)
            
            self.allocationTable[k] = self.opt.actNBits
            local ofr
            for j=1, self.opt.actNBits do
                ofr = utee.overflowRateTable(v, j, self.opt.actNBits - j)
                if ofr <= self.opt.overFlowRate then
                    self.allocationTable[k] = j
                    break
                end
            end
            local layerName = torch.typename(self.shadow:get(k))
            print(("%s, %d.%d, ofr: %.6f")
                :format(layerName, self.allocationTable[k], self.opt.actNBits - self.allocationTable[k], ofr))
        end
    end
end

function Trainer:quantizationForward()
    for i=1, #self.shadow do
        if i == 1 then
            self.shadow:get(i):forward(self.input)
        else
            self.shadow:get(i):forward(self.shadow:get(i-1).output)
        end
        if self.allocationTable[i] then
            local shiftToVal = 2^-self.shiftNBits[i] * self.shadow:get(i).output
            local quantizationVal = utee.quantization(
                    shiftToVal, 
                    self.allocationTable[i],
                    self.opt.actNBits - self.allocationTable[i]
                )
            local shiftBackVal = 2^self.shiftNBits[i] * quantizationVal
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
            print(layerName, meanVal, minVal, maxVal)
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
    if not self.allocationTable then
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
