local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local Avg = cudnn.SpatialAveragePooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
    local model = nn.Sequential()                 
    model:add(Convolution(3, 16, 3, 3, 1, 1, 1, 1))
    model:add(ReLU(true))
    model:add(Max(2, 2, 2, 2))
    model:add(Convolution(16, 16, 3, 3, 1, 1, 1, 1))
    model:add(ReLU(true))
    model:add(Max(2, 2, 2, 2))
    model:add(Convolution(16, 16, 3, 3, 1, 1, 1, 1))
    model:add(ReLU(true))
    model:add(Max(2, 2, 2, 2))
    model:add(Convolution(16, 16, 3, 3, 1, 1, 1, 1))
    model:add(ReLU(true))
    model:add(Max(2, 2, 2, 2))
    model:add(Convolution(16, 16, 3, 3, 1, 1, 1, 1))
    model:add(ReLU(true))
    model:add(Avg(2, 2, 2, 2))
    model:add(nn.View(opt.batchSize, -1))
    model:add(nn.Linear(16*7*7, 1000))
    
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


    if opt.device == 'gpu' then
        model:cuda()
    end
    model:get(1).gradInput = nil
    print(model)
    return model
end

return createModel
