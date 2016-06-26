local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local Avg = cudnn.SpatialAveragePooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
    assert(opt.nUnit % 2 == 0, 'opt.nUnit should be even number')
    
    local function casResUnit(nInputPlane, convType1, convType2)
        -- chunk11
        local chunk11 = nn.Sequential()
        chunk11:add(Convolution(nInputPlane, nInputPlane, 3, 3, 1, 1, 1, 1))
        chunk11:add(SBatchNorm(nInputPlane))
        
        -- chunk12
        local chunk12 = nn.Sequential()
        if convType1 == 'conv1x1' then
            chunk12:add(Convolution(nInputPlane, nInputPlane, 1, 1, 1, 1, 0, 0))
        elseif convType1 == 'conv3x3' then
            chunk12:add(Convolution(nInputPlane, nInputPlane, 3, 3, 1, 1, 1, 1))
        else
            assert(nil, 'Unknown convType1: ' .. convType1)
        end
        chunk12:add(SBatchNorm(nInputPlane))
        chunk12:add(ReLU(true))
        if convType2 == 'conv1x1' then
            chunk12:add(Convolution(nInputPlane, nInputPlane, 1, 1, 1, 1, 0, 0))
        elseif convType2 == 'conv3x3' then
            chunk12:add(Convolution(nInputPlane, nInputPlane, 3, 3, 1, 1, 1, 1))
        else
            assert(nil, 'Unknown convType2: ' .. convType2)
        end
        chunk12:add(SBatchNorm(nInputPlane))

        -- chunk21
        local chunk21 = nn.Sequential()
        chunk21:add(Convolution(nInputPlane, nInputPlane, 3, 3, 1, 1, 1, 1))
        chunk21:add(SBatchNorm(nInputPlane))

        -- chunk22
        local chunk22 = nn.Sequential()
        if convType1 == 'conv1x1' then
            chunk22:add(Convolution(nInputPlane, nInputPlane, 1, 1, 1, 1, 0, 0))
        elseif convType1 == 'conv3x3' then
            chunk22:add(Convolution(nInputPlane, nInputPlane, 3, 3, 1, 1, 1, 1))
        else
            assert(nil, 'Unknown convType1: ' .. convType1)
        end
        chunk22:add(SBatchNorm(nInputPlane))
        chunk22:add(ReLU(true))
        if convType2 == 'conv1x1' then
            chunk22:add(Convolution(nInputPlane, nInputPlane, 1, 1, 1, 1, 0, 0))
        elseif convType2 == 'conv3x3' then
            chunk22:add(Convolution(nInputPlane, nInputPlane, 3, 3, 1, 1, 1, 1))
        else
            assert(nil, 'Unknown convType2: ' .. convType2)
        end
        chunk22:add(SBatchNorm(nInputPlane))
   
        -- s1, s2, exParams
        s1 = nn.Sequential()
        aux11 = nn.Sequential()
        aux11:add(chunk11)
        aux11:add(ReLU(true))
        aux11:add(chunk21)
        s1:add(nn.ConcatTable():add(aux11):add(nn.Identity()))
        s1:add(nn.CAddTable())
        s1:add(ReLU(true))
        
        s2 = nn.Sequential()
        aux21 = nn.Sequential()
        aux21:add(nn.ConcatTable():add(chunk11):add(chunk12))
        aux21:add(nn.CAddTable())
        aux21:add(ReLU(true))
        aux21:add(nn.ConcatTable():add(chunk21):add(chunk22))
        aux21:add(nn.CAddTable())
        s2:add(nn.ConcatTable():add(aux21):add(nn.Identity()))
        s2:add(nn.CAddTable())
        s2:add(ReLU(true))
        
        exParams = nn.Sequential()
        exParams:add(chunk12)
        exParams:add(chunk22)
        return s1, s2, exParams
    end

    local function transitionUnit(nInputPlane, nOutputPlane)
        local convs = nn.Sequential()
        convs:add(Convolution(nInputPlane, nOutputPlane, 3, 3, 2, 2, 1, 1))
        convs:add(SBatchNorm(nOutputPlane))
        convs:add(ReLU(true))
        convs:add(Convolution(nOutputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        convs:add(SBatchNorm(nOutputPlane))  
        shortcut = Convolution(nInputPlane, nOutputPlane, 1, 1, 2, 2, 0, 0)

        local s = nn.Sequential()
        s:add(nn.ConcatTable():add(shortcut):add(convs))
        s:add(nn.CAddTable())
        s:add(ReLU(true))
        return s
    end
    
    local model1 = nn.Sequential()
    local model2 = nn.Sequential()
    local ex = nn.Sequential()
    if opt.dataset == 'cifar10' then
        depth = 3 * opt.nUnit + 2

        local nStages = {3, 16, 32, 64}

        -- first fix layer
        fixChunk = nn.Sequential()
        fixChunk:add(Convolution(nStages[1], nStages[2], 3, 3, 1, 1, 1, 1))
        fixChunk:add(SBatchNorm(nStages[2]))
        fixChunk:add(ReLU(true))
        model1:add(fixChunk)
        model2:add(fixChunk)
        
        -- stage 1
        for i=1, opt.nUnit/2 do
            local s1Tmp, s2Tmp, exParamsTmp = casResUnit(nStages[2], opt.convType1, opt.convType2)
            model1:add(s1Tmp)
            model2:add(s2Tmp)
            ex:add(exParamsTmp)
        end

        -- stage 2
        trans2 = transitionUnit(nStages[2], nStages[3])
        model1:add(trans2)
        model2:add(trans2)
        for i=1, opt.nUnit/2-1 do
            local s1Tmp, s2Tmp, exParamsTmp = casResUnit(nStages[3], opt.convType1, opt.convType2)
            model1:add(s1Tmp)
            model2:add(s2Tmp)
            ex:add(exParamsTmp)
        end

        -- stage 3
        trans2 = transitionUnit(nStages[3], nStages[4])
        model1:add(trans2)
        model2:add(trans2)
        for i=1, opt.nUnit/2-1 do
            local s1Tmp, s2Tmp, exParamsTmp = casResUnit(nStages[4], opt.convType1, opt.convType2)
            model1:add(s1Tmp)
            model2:add(s2Tmp)
            ex:add(exParamsTmp)
        end
        
        -- output
        outputChunk = nn.Sequential()
        outputChunk:add(Avg(8, 8, 1, 1))
        outputChunk:add(nn.View(nStages[4]):setNumInputDims(3))
        outputChunk:add(nn.Linear(nStages[4], 10))
        model1:add(outputChunk)
        model2:add(outputChunk)
    else
        error('invalid dataset: ' .. opt.dataset)
    end

    local function ConvInit(model, name)
        for k,v in pairs(model:findModules(name)) do
            local n = v.kW*v.kH*v.nOutputPlane
            v.weight:normal(0,math.sqrt(2/n))
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
            v.weight:uniform(0, math.sqrt(4 / n))
            v.bias:fill(0)
        end
    end
    
    ConvInit(model1, 'cudnn.SpatialConvolution')
    ConvInit(model1, 'nn.SpatialConvolution')
    BNInit(model1, 'fbnn.SpatialBatchNormalization')
    BNInit(model1, 'cudnn.SpatialBatchNormalization')
    BNInit(model1, 'nn.SpatialBatchNormalization')
    LinearInit(model1, 'nn.Linear')
    
    ConvInit(model2, 'cudnn.SpatialConvolution')
    ConvInit(model2, 'nn.SpatialConvolution')
    BNInit(model2, 'fbnn.SpatialBatchNormalization')
    BNInit(model2, 'cudnn.SpatialBatchNormalization')
    BNInit(model2, 'nn.SpatialBatchNormalization')
    LinearInit(model2, 'nn.Linear')
    
    model1:cuda()
    model2:cuda()

    model1:get(1).gradInput = nil
    model2:get(1).gradInput = nil

    print(model1)
    print(model2)
    print('Cascade-Resnet-' .. depth .. ' CIFAR-10')
    return model1, model2, ex
end

return createModel
