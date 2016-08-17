require 'loadcaffe'

local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local Avg = cudnn.SpatialAveragePooling
local SBatchNorm = nn.SpatialBatchNormalization
local utee = require 'utee'

local function createModel(opt)
    local model
    -- load torch model first if exists
    if  paths.filep(opt.torchModelPath) then
        print("loading torch model from " .. opt.torchModelPath)
        model = torch.load(opt.torchModelPath)
    else
        assert(opt.device == 'cpu', 'Support cpu only')
        print("loading caffe model from " .. opt.modelPath)
        model = loadcaffe.load(opt.netPath, opt.modelPath, 'nn')
        print("Exchanging the first kernel order from BGR to RGB")
        for i=1, #model do
            layerName = torch.typename(model:get(i))
            if layerName == 'nn.SpatialConvolution' or layerName == 'cudnn.SpatialConvolution' then
                print("- Find " .. layerName .. ', exchange it')
                weight = model:get(i).weight
                tmp = weight[{{}, {1}, {}, {}}]:clone()
                weight[{{}, {1}, {}, {}}] = weight[{{}, {3}, {}, {}}]:clone()
                weight[{{}, {3}, {}, {}}] = tmp
                break
            end
        end
    end

    -- remove softmax for efficiency
    local lastLayerName = torch.typename(model:get(#model))
    if lastLayerName == 'nn.SoftMax' or lastLayerName == 'cudnn.SoftMax' then
        if opt.testOnly then
            print(("Removing last layer %s"):format(lastLayerName))
            model:remove(#model)
        end
    end

    print('Substituting SpatialConvolution with SpationConvolutionFixedPoint')
    for i=1,#model do
        local layerName = torch.typename(model:get(i))
        if layerName == 'nn.SpatialConvolution' then
            local tmp = model:get(i):clone()
            model:remove(i)
            model:insert(utee.substitute(tmp), i)
        end
    end

    -- remove inplace
    for i=1, #model do
        if model:get(i).inplace then
            model:get(i).inplace = false
        end
    end

    model:clearState()
    return model

end

return createModel
