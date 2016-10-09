require 'loadcaffe'

local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local Avg = cudnn.SpatialAveragePooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
    local model
    -- load torch model first if exists
    if paths.filep(opt.torchModelPath) then
        print("loading torch model from " .. opt.torchModelPath)
        model = torch.load(opt.torchModelPath)
    else
        local loadType = opt.device == 'gpu' and 'cudnn' or opt.device == 'cpu' and 'nn' or nil
        assert(loadType, 'Neither gpu nor cpu')
        print("loading caffe model from " .. opt.modelPath)
        model = loadcaffe.load(opt.netPath, opt.modelPath, loadType)
        opt.swapChannel = true
    end

    if opt.swapChannel then
        for i=1, #model do
            layerName = torch.typename(model:get(i))
            local m = model:get(i)
            if layerName == 'nn.SpatialConvolution' 
                or layerName == 'nn.SpatialConvolutionMM'
                or layerName == 'cudnn.SpatialConvolution' then
                if #m.weight:size() ~= 4 then
                    print("Reshape nn.SpatialConvolutionMM")
                    m.weight = torch.reshape(m.weight, m.nOutputPlane, m.nInputPlane, m.kH, m.kW)
                end
                print("Exchanging the first kernel order from BGR to RGB")
                print("- Find " .. layerName .. ', exchange it')
                weight = m.weight
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
    
    -- remove the last block
    local lastLayerName = torch.typename(model:get(#model))
    if lastLayerName == 'nn.Linear' then
        if opt.testOnly then
            print(("Removing last layer %s"):format(lastLayerName))
            model:remove(#model)
            model:remove(#model)
            model:remove(#model)
        end
    end
    print(model)

    
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
