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
    local loadType = opt.device == 'gpu' and 'cudnn' or opt.device == 'cpu' and 'nn' or nil
    assert(loadType, 'Neither gpu nor cpu')
    local nClasses = opt.nClasses

    model = loadcaffe.load(netPath, modelPath, loadType)

    if opt.device == 'gpu' then model:cuda() end

    if opt.cudnn == 'deterministic' then
        model:apply(function(m)
                if m.setMode then model:setMode(1,1,1) end
            end)
    end

    model:get(1).gradInput = nil
    
    -- remove inplace
    for i=1, #model do
        if model:get(i).inplace then
            model:get(i).inplcae = false
        end
    end
    print(model)
    return model
end

return createModel
