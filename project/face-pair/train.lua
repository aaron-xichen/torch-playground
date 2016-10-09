local optim = require 'optim'
local utee = require 'utee'

local M = {}
local Trainer = torch.class('facePair.Trainer', M)

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
                    if self.opt.shiftInfoTable then
                        self.orders[i].weightShiftBits = self.opt.shiftInfoTable[i][1]
                        self.orders[i].biasShiftBits = self.opt.shiftInfoTable[i][2]
                    else
                        self.orders[i].weightShiftBits = -torch.ceil(torch.log(torch.abs(weight):max()) / torch.log(2))
                        self.orders[i].biasShiftBits = -torch.ceil(torch.log(torch.abs(bias):max()) / torch.log(2))
                    end

                    local paramNBits
                    if layerName == 'cudnn.SpatialConvolution' 
                        or layerName == 'nn.SpatialConvolutionMM'
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
                        local weightShiftBits = self.orders[i].weightShiftBits
                        weight:copy(2^-weightShiftBits * utee.quantization(weight * 2^weightShiftBits, 1, paramNBits-1))
                    end

                    if torch.abs(bias):max() ~= 0 then
                        local biasShiftBits = self.orders[i].biasShiftBits
                        bias:copy(2^-biasShiftBits * utee.quantization(bias * 2^biasShiftBits, 1, paramNBits-1))
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
        elseif self.opt.device == 'cpu' and self.opt.tensorType == 'float' then
            self.orders[i]:type('torch.FloatTensor')
        elseif self.opt.device == 'gpu' then
            self.orders[i]:type('torch.CudaTensor')
        end
        if self.orders[i].weight then
            print(torch.typename(self.orders[i].weight))
        end
    end
end

function Trainer:analyzeAct()
    if self.opt.actNBits ~= -1 then
        print("=> Analyzing activation distribution")
        if self.opt.shiftInfoTable then
            for i=1, #self.orders do
                if self.opt.shiftInfoTable[i] then
                    self.orders[i].actShiftBits = self.opt.shiftInfoTable[i][3]
                end
            end
        else
            for n, sample in self.valDataLoader:run() do
                local input1, input2 = sample.input1, sample.input2
                local output1, output2
                if self.opt.device == 'gpu' then 
                    input1 = input1:cuda()
                    input2 = input2:cuda()
                end
                utee.analyzeAct(self.model, input1, self.opt)
                utee.analyzeAct(self.model, input2, self.opt)
                if n >= self.opt.collectNSamples then
                    self.valDataLoader:reset()
                    break
                end
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

function Trainer:quantizationForward(input)
    if self.opt.device == 'cpu' and self.opt.tensorType == 'double' then
        utee.quantizationForward(self.model, input:double(), self.opt.actNBits, self.opt.debug)
    else 
        utee.quantizationForward(self.model, input, self.opt.actNBits, self.opt.debug)
    end
    return self.model:get(#self.model).output
end

function Trainer:manualForward(input)
    if self.opt.device == 'cpu' and self.opt.tensorType == 'double' then
        self.model:forward(input:double())
    else 
        self.model:forward(input)
    end
    return self.model:get(#self.model).output
end

function Trainer:val()
    local size = self.valDataLoader:size()
    local top1Sum, top5Sum = 0.0, 0.0
    local N = 0

    self:quantizeParam()
    self:castTensorType()
    self:analyzeAct()

    shiftTable = {}
    for k, v in pairs(self.orders) do
        if v.weightShiftBits then
            print(k, v.weightShiftBits, v.biasShiftBits, v.actShiftBits)
            shiftTable[k] = {v.weightShiftBits, v.biasShiftBits, v.actShiftBits}
        end
    end
    
    --print("Saving shift info to " .. self.opt.shiftInfoSavePath)
    --torch.save(self.opt.shiftInfoSavePath, shiftTable)

    -- forward
    local pairList, nEpoch = {}, self.valDataLoader:size()
    for n, sample in self.valDataLoader:run() do 
        local input1, input2 = sample.input1, sample.input2
        local output1, output2
        if self.opt.device == 'gpu' then 
            input1 = input1:cuda()
            input2 = input2:cuda()
        end
        --print(('data: %.3f, %.3f'):format(torch.mean(torch.abs(input1)), torch.mean(torch.abs(input2))))
        if self.opt.actNBits == -1 then
            output1 = self:manualForward(input1):clone()
            output2 = self:manualForward(input2):clone()
        else
            output1 = self:quantizationForward(input1):clone()
            output2 = self:quantizationForward(input2):clone()
        end

        output1 = torch.reshape(output1, self.opt.batchSize, self.opt.crop, output1:size()[2]):mean(2):squeeze()
        output2 = torch.reshape(output2, self.opt.batchSize, self.opt.crop, output2:size()[2]):mean(2):squeeze()

        local score = torch.norm(output1-output2, 2, 2):view(-1):totable()
        for i=1, #score do
            table.insert(pairList, {score[i], sample.target[i]})
        end
        print(("%d/%d"):format(n, nEpoch))
    end
    pairList = torch.FloatTensor(pairList)
    torch.save(self.opt.savePath, pairList)
    print(("=> Quantization Info, ConvParam: %d bits, FcParam: %d bits, Activation: %d bits")
        :format(self.opt.convNBits, self.opt.fcNBits, self.opt.actNBits))
    vals = {0.0, 0.0}
    return vals
end

------------ helper function ----------------
function Trainer:train(epoch)
    assert(nil, 'Not Implement')
end

return M.Trainer
