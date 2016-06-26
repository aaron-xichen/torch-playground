local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local Avg = cudnn.SpatialAveragePooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
    local function fracC3(nInputPlane, nOutputPlane)
        local right1 = nn.Sequential()
        :add(Convolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        :add(SBatchNorm(nOutputPlane))
        :add(ReLU(true))
        :add(Convolution(nOutputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        :add(SBatchNorm(nOutputPlane))
        :add(ReLU(true))
        
        local right2 = nn.Sequential()
        :add(Convolution(nOutputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        :add(SBatchNorm(nOutputPlane))
        :add(ReLU(true))
        :add(Convolution(nOutputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        :add(SBatchNorm(nOutputPlane))
        :add(ReLU(true))

        local mid1 = nn.ConcatTable()
        :add(
            nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
            :add(SBatchNorm(nOutputPlane))
            :add(ReLU(true)))
        :add(right1)
        
        local mid2 = nn.ConcatTable()
        :add(
            nn.Sequential()
            :add(Convolution(nOutputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
            :add(SBatchNorm(nOutputPlane))
            :add(ReLU(true)))
        :add(right2)
        
        local high = nn.Sequential()
        :add(
            nn.ConcatTable()
            :add(
                nn.Sequential()
                :add(Convolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
                :add(SBatchNorm(nOutputPlane))
                :add(ReLU(true))) 
            :add(
                nn.Sequential()
                :add(mid1)
                :add(nn.CAddTable())
                :add(nn.MulConstant(1 / 2))
                :add(mid2)
                :add(nn.CAddTable())))
        :add(nn.CAddTable())
        :add(nn.MulConstant(1 / 3))
        
        return high
    end    

    local model = nn.Sequential()
    if opt.dataset == 'cifar10' then
        -- local nStages = {3, 64, 128, 256, 512, 512}
        local nStages = {3, 16, 32, 64, 128, 128}

        for i=1,4 do   
            model:add(fracC3(nStages[i], nStages[i+1]))
            for j=1,opt.nUnit-1 do
                model:add(fracC3(nStages[i+1], nStages[i+1]))
            end
            model:add(Max(2, 2, 2, 2))
        end
        model:add(fracC3(nStages[5], nStages[6]))
        for j=1, opt.nUnit-1 do
            model:add(fracC3(nStages[5], nStages[6]))
        end
        model:add(Avg(2, 2, 2, 2))

        model:add(nn.View(nStages[6]):setNumInputDims(3))
        model:add(nn.Linear(nStages[6], 10))
    else
        error('invalid dataset: ' .. opt.dataset)
    end

    local function ConvInit(model, name)
        for k,v in pairs(model:findModules(name)) do
            local n = v.kW*v.kH*v.nOutputPlane
            v.weight:normal(0, math.sqrt(2/n))
            if cudnn.version >= 4000 then
                v.bias = nil
                v.gradBias = nil
            else
                v.bias:zero()
            end
        end
    end
    
    local function BNInit(model, name)
        for k,v in pairs(model:findModules(name)) do
            v.weight:fill(1)
            v.bias:zero()
        end
    end
    local function LinearInit(model, name)
        for k, v in pairs(model:findModules(name)) do
            local n = v.weight:size(1) + v.weight:size(2)
            v.weight:normal(0, math.sqrt(4 / n))
            v.bias:fill(0)
        end
    end

    ConvInit(model, 'cudnn.SpatialConvolution')
    ConvInit(model, 'nn.SpatialConvolution')
    BNInit(model, 'fbnn.SpatialBatchNormalization')
    BNInit(model, 'cudnn.SpatialBatchNormalization')
    BNInit(model, 'nn.SpatialBatchNormalization')
    LinearInit(model, 'nn.Linear')

    model:cuda()
    model:get(1).gradInput = nil
    print(model)
    print('fracNetC3B5 CIFAR-10')
    return model
end

return createModel
