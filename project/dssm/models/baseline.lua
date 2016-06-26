require 'project/dssm/CicularShift';
local function createModel(opt)

    -- handle args
    if opt.activation == 'ReLU' then
        actFunc = nn.ReLU
    elseif opt.activation == 'Tanh' then
        actFunc = nn.Tanh
    else
        assert(nil, 'Unknown activation ' .. opt.activation)
    end
    nFeature = opt.nFeature
    batchSize = opt.batchSize
    offsetRatio = opt.offsetRatio
    nTrial = opt.nTrial

    -- build model
    model = nn.Sequential()
    if opt.sparse then
        sparse = nn.Sequential():add(nn.SparseLinear(nFeature + 1, 400)):add(nn.View(1, 400))
        model:add(sparse)
        model:add(nn.JoinTable(1))
    else
        model:add(nn.Linear(nFeature, 400))
    end

    model:add(actFunc())
    model:add(nn.Linear(400, 120))
    model:add(actFunc())
    model:add(nn.View(-1, batchSize, 120))
    model:add(nn.SplitTable(1))

    par2 = nn.ParallelTable()

    concat1 = nn.Concat(1)
    for i=1, nTrial+1 do
        concat1:add(nn.Identity())
    end
    concat2 = nn.Concat(1)
    concat2:add(nn.Identity())
    for i=1, nTrial do
        base = math.floor(offsetRatio * batchSize)
        range = math.floor(0.8 * batchSize)
        offset = base + torch.random(range)
        concat2:add(nn.CicularShift(1, offset, batchSize))
    end
    par2:add(concat1)
    par2:add(concat2)
    model:add(par2)
    model:add(nn.CosineDistance())
    model:add(nn.View(-1, batchSize))
    model:add(nn.Transpose({1, 2}))
    model:add(nn.MulConstant(20))

    -- weights initialization
    local function LinearInit(name)
        for k, v in pairs(model:findModules(name)) do
            local n = v.weight:size(1) + v.weight:size(2)
            print('Initializing ' .. name)
            v.weight:uniform(-math.sqrt(6/n), math.sqrt(6/n))
            v.bias:zero()
        end
    end
    LinearInit('nn.Linear')
    LinearInit('nn.SparseLinear')
    
    -- enable gpu
    model:cuda()

    if opt.cudnn == 'deterministic' then
        model:apply(function(m)
                if m.setMode then model:setMode(1,1,1) end
            end)
    end

    model:get(1).gradInput = nil
    print('DSSM Training')
    return model
end

return createModel
