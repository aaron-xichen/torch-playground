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

    --[[
    self.criterion = criterion
    self.optimState = optimState or {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    nesterov = true,
    dampening = 0.0,
    weightDecay = opt.weightDecay,
}
    self.params, self.gradParams = self.shadow:getParameters()
    --]]

    self.opt = opt

    self.trainDataLoader = trainDataLoader
    self.valDataLoader = valDataLoader

    self.collectNSamples = 10
end

function Trainer:paramQuantization()
    if self.opt.convNBits ~= -1 and self.opt.fcNBits ~= -1 then
        print('=> Quantizing weights and bias') 
        self.paramShiftNBits = {}
        local traceParam = {}
        for i=1, #self.model do
            if self.model:get(i).weight then
                local weight = self.model:get(i).weight
                local bias = self.model:get(i).bias
                local layerName = torch.typename(self.model:get(i))
                local paramNBits
                if layerName == 'cudnn.SpatialConvolution' 
                    or layerName == 'nn.SpatialConvolution' 
                    or layerName == 'nn.SpatialConvolutionFixedPoint' then
                    paramNBits = self.opt.convNBits
                elseif layerName == 'nn.Linear' then
                    paramNBits = self.opt.fcNBits
                else
                    assert(nil, "Unknow layer type " .. layerName)
                end

                self.paramShiftNBits[i] = {}
                self.paramShiftNBits[i][1] = -torch.ceil(torch.log(torch.abs(weight):max()) / torch.log(2))
                self.paramShiftNBits[i][2] = -torch.ceil(torch.log(torch.abs(bias):max()) / torch.log(2))

                local weightShiftTo = weight * 2 ^ self.paramShiftNBits[i][1]
                local weightQuantization = utee.quantization(weightShiftTo, 1, paramNBits - 1)
                local weightShiftBack = weightQuantization * 2 ^ -self.paramShiftNBits[i][1]
                self.model:get(i).weight:copy(weightShiftBack)

                local biasShiftTo = bias * 2 ^ self.paramShiftNBits[i][2]
                local biasQuantization = utee.quantization(biasShiftTo, 1, paramNBits - 1)
                local biasShiftBack = biasQuantization * 2 ^ -self.paramShiftNBits[i][2]
                self.model:get(i).bias:copy(biasShiftBack)

                print(layerName)
                print('weight: ', self.paramShiftNBits[i][1], 'bias: ', self.paramShiftNBits[i][2])

                table.insert(traceParam, self.model:get(i).weight:float())
                table.insert(traceParam, self.model:get(i).bias:float())

            end
        end

        --torch.save('cpuParam.t7', traceParam)
        print(("=> Quantization Info, ConvParam: %d bits, FcParam: %d bits, Activation: %d bits")
            :format(self.opt.convNBits, self.opt.fcNBits, self.opt.actNBits))
    end
end

function Trainer:castTensorType()
    for i=1, #self.model do
        if self.opt.device == 'cpu' and self.opt.tensorType == 'double' then
            self.model:get(i):type('torch.DoubleTensor')
        end
        if self.model:get(i).weight then
            print(torch.typename(self.model:get(i).weight))
        end
    end
end

function Trainer:actAnalysis()
    if self.opt.actNBits ~= -1 then
        print("=> Analyzing activation distribution")
        --[[
        local cache = {}

        print(('=> Sampling %d data points'):format(self.collectNSamples))
        for n, sample in self.valDataLoader:run() do
        self:copyInputs(sample)
        self.model:forward(self.input)
        for i=1, #self.model do
        local layerName = torch.typename(self.model:get(i))
        if self.relatedLayers[layerName] then
        if not cache[i] then
        cache[i] = {}
    end
        table.insert(cache[i], self.model:get(i).output)
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
        print(k, torch.typename(self.model:get(k)), self.actShiftNBits[k])
    end]]--
        self.actShiftNBits = {
            [1] = -10,
            [3] = -12,
            [6] = -13,
            [8] = -14,
            [11] = -15,
            [13] = -14,
            [15] = -15,
            [18] = -14,
            [20] = -13,
            [22] = -12,
            [25] = -12,
            [27] = -11,
            [29] = -10,
            [33] = -7,
            [36] = -5,
            [39] = -6
        }
        print('=> Analyzing done!')
    end
end

function Trainer:quantizationForward()
    for i=1, #self.model do
        if i == 1 then
            if self.opt.device == 'cpu' and self.opt.tensorType == 'double' then
                self.model:get(i):forward(self.input:double())
            else 
                self.model:get(i):forward(self.input)
            end
        else
            self.model:get(i):forward(self.model:get(i-1).output)
        end

        if self.actShiftNBits[i] then
            --print(i)
            local output = self.model:get(i).output
            --local outputReal = output:float()
            --print(outputReal:sum(), outputReal:min(), outputReal:max())
            local shiftToVal = 2^self.actShiftNBits[i] * output
            local quantizationVal = utee.quantization(
                shiftToVal, 
                1,
                self.opt.actNBits - 1
            )
            local shiftBackVal = 2^-self.actShiftNBits[i] * quantizationVal

            output:copy(shiftBackVal)

            --outputReal = output:float()
            --print(outputReal:sum(), outputReal:min(), outputReal:max())  
        end

    end
    --[[
    traceOutput = {}
    traceParam = {}
    table.insert(traceOutput, self.input:float())
    for i = 1, #self.model do
    if self.model:get(i).weight then
    table.insert(traceParam, self.model:get(i).weight:float())
end
    if self.model:get(i).bias then
    table.insert(traceParam, self.model:get(i).bias:float())
end
    table.insert(traceOutput, self.model:get(i).output:float())
end
    torch.save('traceOutput.t7', traceOutput)
    torch.save('traceParam.t7', traceParam)
    ]]--
end



function Trainer:val()
    local size = self.valDataLoader:size()
    local nCrops = self.opt.tenCrop and 10 or 1
    local top1Sum, top5Sum = 0.0, 0.0
    local N = 0

    --self.valDataLoader:reset()

    -- init
    self:paramQuantization()
    self:castTensorType()
    if not self.actShiftNBits then
        self:actAnalysis()
    end

    self.model:evaluate()
    --cutorch.manualSeed(20)
    
   -- torch.manualSeed(100)
    -- forward
    for n, sample in self.valDataLoader:run() do
        self:copyInputs(sample)
        torch.save('input.t7', sample.input:float())
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

        --self.valDataLoader:reset()
        --break
        --if N >= 1 then break end
    end

    print((' * Val Done, Top1Err: %7.3f  Top5Err: %7.3f'):format(top1Sum / N, top5Sum / N))
    print(("=> Quantization Info, ConvParam: %d bits, FcParam: %d bits, Activation: %d bits")
        :format(self.opt.convNBits, self.opt.fcNBits, self.opt.actNBits))
    vals = {top1Sum / N, top5Sum / N}
    return vals
end

------------ helper function ----------------
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

function Trainer:learningRate(epoch)
    decay = epoch >= 375 and 4 or epoch >= 350 and 3 or epoch >= 300 and 2 or epoch >=200 and 1 or 0
    return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
