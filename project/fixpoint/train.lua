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
    self.model:evaluate()
    self.opt = opt
    self.trainDataLoader = trainDataLoader
    self.valDataLoader = valDataLoader
end

function Trainer:fillParamInt32()
    print('=> Filling prameters in INT32 format')  

    for i=1, #self.model do
        if self.model:get(i).weight then
            local meta = self.opt.metaTable[i]
            local config = self.opt.bitWidthConfig[i]
            local weight = self.model:get(i).weight:clone()
            local bias = self.model:get(i).bias:clone()

            local weightShiftBits, biasShiftBits, biasAlignShiftBits = meta[1], meta[2], meta[4]
            local weightBitWidth, biasBitWidth = config[1], config[2]

            -- fixed point weight
            local weight1 = utee.fixedPoint(weight * 2^weightShiftBits, 1, weightBitWidth-1)
            self.model:get(i).weight:copy(weight1)

            -- fixed point bias and align
            local bias1 = utee.fixedPoint(bias * 2^biasShiftBits, 1, biasBitWidth-1) * 2^biasAlignShiftBits
            self.model:get(i).bias:copy(bias1)
        end
    end
end

function Trainer:castToInt32Type()
    for i=1, #self.model do
        self.model:get(i):type('torch.IntTensor')
    end
    print("Tensor type: ", torch.typename(self.model:get(1).weight))
end

function Trainer:forwardInt32()
    for i=1, #self.model do
        local layerName = torch.typename(self.model:get(i))
        if i == 1 then
            self.model:get(i):forward(self.input)
        else
            self.model:get(i):forward(self.model:get(i-1).output)
        end
        local meta = self.opt.metaTable[i]
        if meta then
            local winShiftBits, decPosSave, decPosRaw = meta[5], meta[6], meta[7]
            local actBitWidth = self.opt.bitWidthConfig[i][3]

            local output = self.model:get(i).output
            --[[
            local outputTmp1 = output:float() * 2^decPosRaw
            print(outputTmp1:sum(), outputTmp1:min(), outputTmp1:max())
            ]]--

            local maxVal = 2^(actBitWidth-1)-1
            local shiftLeft = bit.lshift(maxVal, winShiftBits)
            local overflow = bit.lshift(maxVal+1, winShiftBits) 
            local roundBit = bit.lshift(0x1, winShiftBits-1)
            local sign = torch.sign(output)

            output:abs():apply(
                function(x)
                    if bit.band(x, overflow) ~= 0 then -- overflow, return max
                        return maxVal
                    elseif bit.band(x, roundBit) ~= 0 then -- ceil
                        return math.min(bit.rshift(bit.band(x, shiftLeft), winShiftBits) + 1, maxVal)
                    else -- floor
                        return bit.rshift(bit.band(x, shiftLeft), winShiftBits)
                    end
                end
            )
            output:cmul(sign)

            
            --[[
            local outputTmp2 = output:float() * 2^decPosSave
            print(outputTmp2:sum(), outputTmp2:min(), outputTmp2:max())
            ]]--
        end

    end
end

function Trainer:val()
    local size = self.valDataLoader:size()
    local nCrops = self.opt.tenCrop and 10 or 1
    local top1Sum, top5Sum = 0.0, 0.0
    local N = 0

    self:fillParamInt32()
    self:castToInt32Type()

    local totalTimer = torch.Timer()
    local timer = torch.Timer()
    timer = torch.Timer()
    for n, sample in self.valDataLoader:run() do
        if N == 0 then
            print("saving to input.t7")
            torch.save('input.t7', sample.input)
        end
        print('data: ', torch.sum(torch.abs(sample.input)))
        timer:reset()
        self:copyInputs(sample)

        self:forwardInt32()

        local top1, top5 = self:computeScore(self.model:get(#self.model).output, sample.target, nCrops)
        top1Sum = top1Sum + top1
        top5Sum = top5Sum + top5
        N = N + 1

        print((' | Val [%d/%d] Top1Err: %7.5f, Top5Err: %7.5f, cost: %3.3fs'):format(n, size, top1, top5, timer:time().real))
        if self.opt.stopNSamples ~= -1 and N >= self.opt.stopNSamples then
            break
        end
    end
    elapse = timer:time().real
    print(("Time elapsed: %3.3f, FPS: %2.2f"):format(elapse, N/elapse))
    
    print((' * Val Done, Top1Err: %7.3f  Top5Err: %7.3f, cost: %3.3fs'):format(top1Sum / N, top5Sum / N, totalTimer:time().real))
    vals = {top1Sum / N, top5Sum / N}
    return vals
end

----------------- helper function ---------------------------------
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
    self.input = sample.input
    self.target = sample.target
end

return M.Trainer
