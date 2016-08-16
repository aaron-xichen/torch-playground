local optim = require 'optim'
local utee = require 'utee'

local M = {}
local Trainer = torch.class('fixpoint.Trainer', M)

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

    self.collectNSamples = 10
end

function Trainer:fillParamInt32()
    print('=> Filling prameters in INT32 format')  
    local layerIdx = 0
    local winShiftBits = 0
    local decPosRaw = 0
    local decPosSave = 0
    self.opt.winShiftTable = {}
    self.opt.decPosRawTable = {}
    self.opt.decPosSaveTable = {}
    self.opt.biasAlignTable = {}

    for i=1, #self.shadow do
        if self.shadow:get(i).weight then
            local weight = self.shadow:get(i).weight:clone()
            local bias = self.shadow:get(i).bias:clone()
            layerIdx = layerIdx + 1

            local weightShiftBits = self.opt.shiftTable[layerIdx][1]
            local biasShiftBits = self.opt.shiftTable[layerIdx][2]
            local actShiftBits = self.opt.shiftTable[layerIdx][3]

            -- fixed point weight
            local weight1 = utee.fixedPoint(weight * 2^weightShiftBits, 1, 7)
            self.shadow:get(i).weight:copy(weight1)

            -- compute shift bits and decimal position
            decPosRaw = decPosSave - 7 - weightShiftBits
            winShiftBits = - (decPosRaw + actShiftBits) - 7
            biasAlignShiftBits = - decPosSave - biasShiftBits + weightShiftBits
            decPosSave = - actShiftBits - 7

            -- fixed point bias and align
            local bias1 = utee.fixedPoint(2^biasShiftBits * bias, 1, 7) * 2^biasAlignShiftBits
            self.shadow:get(i).bias:copy(bias1)

            -- save window shift bits
            self.opt.winShiftTable[i] = winShiftBits
            self.opt.decPosRawTable[i] = decPosRaw
            self.opt.decPosSaveTable[i] = decPosSave
            self.opt.biasAlignTable[i] = biasAlignShiftBits

        end
        self.shadow:get(i):type('torch.IntTensor')
    end

    -- save meta info to disk
    metaInfo = {}
    for k, v in pairs(self.opt.winShiftTable) do
        metaInfo[k] = {}
        table.insert(metaInfo[k], self.opt.shiftTable[k][1]) -- weight
        table.insert(metaInfo[k], self.opt.shiftTable[k][2]) -- bias
        table.insert(metaInfo[k], self.opt.shiftTable[k][3]) -- activation
        table.insert(metaInfo[k], self.opt.biasAlignTable[k]) -- biasAlign
        table.insert(metaInfo[k], self.opt.winShiftTable[k]) -- window shift
        table.insert(metaInfo[k], self.opt.decPosSaveTable[k]) -- decPosSave
        table.insert(metaInfo[k], self.opt.decPosRawTable[k]) -- decPosRaw
    end
    
    print(metaInfo)
    print('Saving meta info to ' .. self.opt.metaInfoPath)
    torch.save(self.opt.metaInfoPath, metaInfo)
    
    assert(layerIdx == #self.opt.shiftTable, 
        ('Layer number does not match, %d vs %d'):format(layerIdx, #self.opt.shiftTable))
end

function Trainer:forwardInt32()
    local decimalPosition = 0
    for i=1, #self.shadow do
        local layerName = torch.typename(self.shadow:get(i))
        if i == 1 then
            self.shadow:get(i):forward(self.input)
        else
            self.shadow:get(i):forward(self.shadow:get(i-1).output)
        end
        if self.opt.winShiftTable[i] then
            print(i)
            local output = self.shadow:get(i).output
            local outputTmp1 = output:float() * 2^self.opt.decPosRawTable[i]
            print(outputTmp1:sum(), outputTmp1:min(), outputTmp1:max())

            local shiftLeft = bit.lshift(0x7f, self.opt.winShiftTable[i])
            local overflow = bit.lshift(0x80, self.opt.winShiftTable[i]) 
            local roundBit = bit.lshift(0x1, self.opt.winShiftTable[i] - 1)
            local sign = torch.sign(output)

           output:abs():apply(
                function(x)
                    -- overflow, return max
                    if bit.band(x, overflow) ~= 0 then 
                        return 127
                        -- ceil
                    elseif bit.band(x, roundBit) ~= 0 then
                        return math.min(bit.rshift(bit.band(x, shiftLeft), self.opt.winShiftTable[i]) + 1, 127)
                        -- floor
                    else
                        return bit.rshift(bit.band(x, shiftLeft), self.opt.winShiftTable[i])
                    end
                end
            )
            output:cmul(sign)

            local outputTmp2 = output:float() * 2^self.opt.decPosSaveTable[i]
            print(outputTmp2:sum(), outputTmp2:min(), outputTmp2:max())
        end

    end
end

function Trainer:val()
    local size = self.valDataLoader:size()
    local nCrops = self.opt.tenCrop and 10 or 1
    local top1Sum, top5Sum = 0.0, 0.0
    local N = 0

    self.shadow:evaluate()
    self.valDataLoader:reset()
    
    self:fillParamInt32()
    
    local totalTimer = torch.Timer()
    local timer = torch.Timer()
    for n, sample in self.valDataLoader:run() do
        print('data: ', torch.sum(sample.input:int():float()))
        print(torch.max(sample.input:int()))
        print(torch.min(sample.input:int()))
        timer:reset()
        self:copyInputs(sample)
        
        self:forwardInt32()

        local top1, top5 = self:computeScore(self.shadow:get(#self.shadow).output, sample.target, nCrops)
        top1Sum = top1Sum + top1
        top5Sum = top5Sum + top5
        N = N + 1

        print((' | Val [%d/%d] Top1Err: %7.5f, Top5Err: %7.5f, cost: %3.3fs'):format(n, size, top1, top5, timer:time().real))
        --self.valDataLoader:reset()
        --break
    end

    print((' * Val Done, Top1Err: %7.3f  Top5Err: %7.3f, cost: %3.3fs'):format(top1Sum / N, top5Sum / N, totalTimer:time().real))
    vals = {top1Sum / N, top5Sum / N}
    return vals
end

----------------- helper function ---------------------------------
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
