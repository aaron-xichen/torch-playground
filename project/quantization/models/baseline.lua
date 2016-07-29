require 'loadcaffe'

local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local Avg = cudnn.SpatialAveragePooling
local SBatchNorm = nn.SpatialBatchNormalization
local utee = require 'utee'

local function createModel(opt)
    local netPath = opt.modelRoot .. '/deploy.prototxt'
    local modelPath = opt.modelRoot .. '/weights.caffemodel'
    local torchModelPath = opt.modelRoot .. '/model.t7'
    
    local model
    
    -- load torch model first if exists
    if  utee.fileExists(torchModelPath) then
        print("loading torch model from " .. torchModelPath)
        model = torch.load(torchModelPath)
    else
        local loadType = opt.device == 'gpu' and 'cudnn' or opt.device == 'cpu' and 'nn' or nil
        assert(loadType, 'Neither gpu nor cpu')
        print("loading caffe model from " .. modelPath)
        model = loadcaffe.load(netPath, modelPath, loadType)
    end

    -- remove softmax for efficiency
    local lastLayerName = torch.typename(model:get(#model))
    if lastLayerName == 'nn.SoftMax' or lastLayerName == 'cudnn.SoftMax' then
        if opt.testOnly then
            print(("Removing last layer %s"):format(lastLayerName))
            model:remove(#model)
        end
    end


    --[[
    print('Substituting SpatialConvolution with SpationConvolutionFixedPoint')
    for i=1,#model do
        local layerName = torch.typename(model:get(i))
        if layerName == 'nn.SpatialConvolution' and opt.device == 'cpu' then
            local tmp = model:get(i):clone()
            model:remove(i)
            model:insert(utee.substitute(tmp), i)
        end
    end
    ]]---

    -- remove inplace
    for i=1, #model do
        if model:get(i).inplace then
            model:get(i).inplace = false
        end
    end

    -- deterministic mode
    model:apply(
        function(m)
            if m.setMode then m:setMode(1,1,1) end
        end
    )

    model:clearState()
    print(model)
    os.exit()
    return model
end

return createModel
