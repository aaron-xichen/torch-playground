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
    self.opt = opt
    self.trainDataLoader = trainDataLoader
    self.valDataLoader = valDataLoader
end

function Trainer:quantizeParam()
    print('=> Quantizing weights and bias') 
    print("Id\tname\tweight\tbias\tact")
    for i=1, #self.model do
        local m = self.model:get(i)
        local layerName = torch.typename(m)
        local weight = m.weight
        local bias = m.bias

        if weight and bias then
            local config = self.opt.bitWidthConfig[i]
            assert(config, ("Bit-width is missing in layer %d"):format(i))
            local weightBitWidth, biasBitWidth, actBitWidth = config[1], config[2], config[3]
            if self.opt.metaTable then
                m.weightShiftBits = self.opt.metaTable[i][1]
                m.biasShiftBits = self.opt.metaTable[i][2]
            else
                m.weightShiftBits = -torch.ceil(torch.log(torch.abs(weight):max()) / torch.log(2))
                m.biasShiftBits = -torch.ceil(torch.log(torch.abs(bias):max()) / torch.log(2))
            end

            if torch.abs(weight):max() ~= 0  then
                local weightShiftBits = m.weightShiftBits
                weight:copy(2^-weightShiftBits * utee.quantization(weight * 2^weightShiftBits, 1, weightBitWidth-1))
            end

            if torch.abs(bias):max() ~= 0 then
                local biasShiftBits = m.biasShiftBits
                bias:copy(2^-biasShiftBits * utee.quantization(bias * 2^biasShiftBits, 1, biasBitWidth-1))
            end

            print(("%d\t%s\t%d\t%d\t%d"):format(i, layerName, weightBitWidth, biasBitWidth, actBitWidth))

        end

    end
    print("<= Quantizing weights and bias done")
end

function Trainer:castTensorType()
    for i=1, #self.model do
        if self.opt.device == 'cpu' then
            self.model:get(i):type('torch.FloatTensor')
        else
            self.model:get(i):type('torch.CudaTensor')
        end
    end
    print(("Tensor type %s"):format(torch.typename(self.model:get(1).weight)))
end

function Trainer:quantizeAct()
    print("=> Analyzing activation distribution dynamic")
    if self.opt.metaTable then
        -- load from meta.config
        for i=1, #self.model do
            if self.opt.metaTable[i] then
                self.model:get(i).actShiftBits = self.opt.metaTable[i][3]
            end
        end
    else
        self.opt.metaTable = {}
        for n, sample in self.valDataLoader:run() do
            self:copyInputs(sample)
            -- fill m.actShiftBits field
            utee.analyzeActDynamic(self.model, self.input, self.opt)
            if n >= self.opt.collectNSamples then
                self.valDataLoader:reset()
                break
            end
        end

        -- generate meta.config
        local decPosRaw = 0
        local decPosSave = 0
        for i=1, #self.model do
            local m = self.model:get(i)
            if m.weight and m.bias then
                local config = self.opt.bitWidthConfig[i]
                assert(config, ("Bit-width is missing in layer %d"):format(i))
                local weightBitWidth, biasBitWidth, actBitWidth = config[1], config[2], config[3]
                local weightShiftBits = m.weightShiftBits
                local biasShiftBits = m.biasShiftBits
                local actShiftBits = m.actShiftBits

                decPosRaw = decPosSave -(weightBitWidth-1) - weightShiftBits
                biasAlignShiftBits = -(biasBitWidth-1) - biasShiftBits - decPosRaw
                decPosSave = -(actBitWidth-1) - actShiftBits 
                winShiftBits = decPosSave - decPosRaw
                
                self.opt.metaTable[i] = {}
                table.insert(self.opt.metaTable[i], weightShiftBits) -- weightShift
                table.insert(self.opt.metaTable[i], biasShiftBits) -- biasShift
                table.insert(self.opt.metaTable[i], actShiftBits) -- actShift
                table.insert(self.opt.metaTable[i], biasAlignShiftBits) -- biasAlign
                table.insert(self.opt.metaTable[i], winShiftBits) -- window shift
                table.insert(self.opt.metaTable[i], decPosSave) -- decPosSave
                table.insert(self.opt.metaTable[i], decPosRaw) -- decPosRaw

            end
        end
    end

    print('<= Analyzing activation distribution dynamic done')
end

function Trainer:quantizationForward()
    utee.quantizationForwardDirectly(self.model, self.input, self.opt)
end


function Trainer:manualForward()
    self.model:forward(self.input)
end

function Trainer:val()
    local size = self.valDataLoader:size()
    local nCrops = self.opt.tenCrop and 10 or 1
    local top1Sum, top5Sum = 0.0, 0.0
    local N = 0
    
    --self:castTensorType()
    local forwardFunc = nil
    -- perform quantization
    if self.opt.bitWidthConfig then
        forwardFunc = self.quantizationForward
        self:quantizeParam()
        self:quantizeAct()
    else
        forwardFunc = self.manualForward
    end
    print(("Tensor type %s"):format(torch.typename(self.model:get(1).weight)))

    -- print and save meta table
    print("Id\tweightShift\tbiasShift\tactShift\tbiasAlign\twinShift\tdecPosSave\tdecPosRaw")
    for k, v in pairs(self.opt.metaTable) do
        local vStr = table.concat(v, "\t")
        local line = tostring(k) .. "\t" .. vStr
        print(line)
    end
    if not self.opt.metaTableExist then
        utee.saveTxt(self.opt.metaTablePath, self.opt.metaTable)
    end

    -- forward
    timer = torch.Timer()
    for n, sample in self.valDataLoader:run() do
        self:copyInputs(sample)
        print('data: ', torch.sum(torch.abs(sample.input)))
        forwardFunc(self)

        local top1, top5 = self:computeScore(self.model:get(#self.model).output, sample.target, nCrops)
        top1Sum = top1Sum + top1
        top5Sum = top5Sum + top5
        N = N + 1
            
        print((' | Val [%d/%d] Top1Err: %7.5f, Top5Err: %7.5f'):format(n, size, top1, top5))
        if self.opt.stopNSamples ~= -1 and N >= self.opt.stopNSamples then
            break
        end
    end
    elapse = timer:time().real
    print(("Time elapsed: %3.3f, FPS: %2.2f"):format(elapse, N/elapse))

    print((' * Val Done, Top1Err: %7.3f  Top5Err: %7.3f'):format(top1Sum / N, top5Sum / N))
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
