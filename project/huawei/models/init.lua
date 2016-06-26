local M = {}

function M.setup(opt)
    local model, optimState, epoch
    if opt.snapshotPath ~= 'none' then
        assert(paths.filep(opt.snapshotPath), 'Saved model not found: ' .. opt.snapshotPath)
        print('=> Resuming model from ' .. opt.snapshotPath)
        snapshot = torch.load(opt.snapshotPath)
        model = snapshot.model
        optimState = snapshot.optimState
        epoch = snapshot.epoch
    else
        local modelPath = 'project/' .. opt.project .. '/models/' .. opt.netType
        print('=> Creating model from file ' .. modelPath .. '.lua')
        model = require(modelPath)(opt)
    end

    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    if opt.cudnn == 'fastest' then
        cudnn.fastest = true
        cudnn.benchmark = true
    elseif opt.cudnn == 'deterministic' then
        model:apply(
            function(m)
                if m.setMode then m:setMode(1, 1, 1) end
            end
        )
    end

    if opt.nGPU > 1 then
        local gpus = torch.range(1, opt.nGPU):totable()
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark

        local dpt = nn.DataParallelTable(1, true, true)
        :add(model, gpus)
        :threads(function()
                local cudnn = require 'cudnn'
                cudnn.fastest, cudnn.benchmark = fastest, benchmark
            end)
        dpt.gradInput = nil

        model = dpt:cuda()
    end

    -- multi-label loss
    local criterion
    if opt.lossWeights ~= 'none' then
        assert(opt.lossWeights:nElement() == opt.nClasses, 
            ('nClasses not match %d vs %d'):format(opt.lossWeights:nElement(), opt.nClasses))
    else
        opt.lossWeights = torch.Tensor(opt.nClasses):fill(1)
    end
    local criterion = nn.BCECriterion(opt.lossWeights)
    
    if opt.loadType == 'cudnn' then criterion:cuda() end

    return model, criterion, optimState, epoch
end

return M
