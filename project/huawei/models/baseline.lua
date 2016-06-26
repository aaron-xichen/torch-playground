require 'loadcaffe'

local function createModel(opt)
    local netPath = opt.modelRoot .. '/deploy.prototxt'
    local modelPath = opt.modelRoot .. '/weights.caffemodel'
    local loadType = opt.loadType
    local nClasses = opt.nClasses
    
    model = loadcaffe.load(netPath, modelPath, loadType)
    -- remove last fc and softmax layers
    model:remove(#model)
    model:remove(#model)
    model:add(nn.Linear(4096, nClasses))
    model:add(nn.Sigmoid(true))
    
    lastFc = model:get(#model - 1)
    local n = lastFc.weight:size(1) + lastFc.weight:size(2)
    lastFc.weight:normal(0, math.sqrt(4 / n))
    lastFc.weight:zero()

    if opt.loadType == 'cudnn' then model:cuda() end

    if opt.cudnn == 'deterministic' then
        model:apply(function(m)
                if m.setMode then model:setMode(1,1,1) end
            end)
    end

    model:get(1).gradInput = nil
    print('Scene Labeling Training')
    print(model)
    return model
end

return createModel
