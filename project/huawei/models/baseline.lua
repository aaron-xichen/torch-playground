require 'loadcaffe'

local function createModel(opt)
    local model
    -- load torch model first if exists
    if  paths.filep(opt.torchModelPath) then
        print("loading torch model from " .. opt.torchModelPath)
        model = torch.load(opt.torchModelPath)
    else
        local loadType = opt.device == 'gpu' and 'cudnn' or opt.device == 'cpu' and 'nn' or nil
        assert(loadType, 'Neither gpu nor cpu')
        print("loading caffe model from " .. opt.modelPath)
        model = loadcaffe.load(opt.netPath, opt.modelPath, loadType)
        print("Exchanging the first kernel order from BGR to RGB")
        for i=1, #model do
            layerName = torch.typename(model:get(i))
            if layerName == 'nn.SpatialConvolution' or layerName == 'cudnn.SpatialConvolution' then
                print("=> Find " .. layerName .. ', exchange it')
                weight = model:get(i).weight
                tmp = weight[{{}, {1}, {}, {}}]:clone()
                weight[{{}, {1}, {}, {}}] = weight[{{}, {3}, {}, {}}]:clone()
                weight[{{}, {3}, {}, {}}] = tmp
                break
            end
        end
    end

    local nClasses = opt.nClasses

    -- remove last fc and softmax layers
    model:remove(#model)
    model:remove(#model)
    
    local lastFC = nn.Linear(4096, nClasses)
    local n = lastFC.weight:size(1) + lastFC.weight:size(2)
    lastFC.weight:normal(0, math.sqrt(4 / n))
    lastFC.bias:zero()
    model:add(lastFC)
    model:add(nn.Sigmoid(true))


    model:get(1).gradInput = nil
    model:clearState()

    if opt.device == 'gpu' then model:cuda() end
    print(model)

    return model
end

return createModel
