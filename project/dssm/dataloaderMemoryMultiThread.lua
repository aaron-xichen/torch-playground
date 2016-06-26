require 'pl';
local t = require 'threads'
local M = {}
local DataLoader = torch.class('dssm.DataLoader', M)
t.serialization('threads.sharedserialize')


function DataLoader.create(dataFilePath, batchSize, nFeature)
    loader = M.DataLoader(dataFilePath, batchSize, nFeature)
    return loader
end

function DataLoader:__init(dataFilePath, batchSize, nFeature)
    self.batchSize = batchSize
    self.nFeature = nFeature

    -- load the t7 file
    timer = torch.Timer()
    print("Loading data from " .. dataFilePath)
    local data = torch.load(dataFilePath)

    assert(data['query']:size()[1] == data['doc']:size()[1], 'Size does not match')
    print(("Loaded query: %d, doc: %d [cost: %f]"):format(data['query']:size()[1], data['doc']:size()[1], timer:time().real))

    self.nSamples = data['query']:size()[1]
    self.nBatch = math.floor(self.nSamples / self.batchSize)

    self.pool = t.Threads(2,
        function(threadid)
            require 'pl'
            require 'cudnn'
            print('starting a new thread ' .. threadid)
        end,
        function(idx)
            _G.query = data['query']
            _G.doc = data['doc']
        end
    )

end 

function DataLoader:size()
    return self.nBatch
end

function DataLoader:nextBatch()
    local pool = self.pool
    local curBatch = 0
    local perm = torch.randperm(self.nSamples)

    local input = nil
    local target = torch.Tensor(self.batchSize):fill(1)

    local function enqueue()
        while curBatch < self.nBatch and pool:acceptsjob() do
            curBatch = curBatch + 1
            pool:addjob(
                function(curBatch, batchSize, nFeature)
                    local data = torch.Tensor(2 * batchSize, nFeature)

                    local beginIdx = batchSize * (curBatch - 1) + 1
                    for j=1, batchSize do
                        local idx = perm[beginIdx + j - 1]

                        for i=1, _G.query:size()[2] do
                            key = _G.query[idx][i][1]
                            val = _G.query[idx][i][2]
                            if key == 0 then break end
                            data[j][key] = val
                        end

                        for i=1, _G.doc:size()[2] do
                            key = _G.doc[idx][i][1]
                            val = _G.doc[idx][i][2]
                            if key == 0 then break end
                            data[j + batchSize][key] = val
                        end
                    end
                    collectgarbage()
                    return data
                end,
                function(_data_)
                    input = _data_
                end,
                curBatch,
                self.batchSize,
                self.nFeature
            )
        end

    end
    function loop()
        timer = torch.Timer()
        enqueue()
        -- print(("stage1: %.6f"):format(timer:time().real))
        timer:reset()
        if not pool:hasjob() then
            return nil
        end
        -- print(("stage2: %.6f"):format(timer:time().real))
        timer:reset()

        pool:dojob()
        -- print(("stage3: %.6f"):format(timer:time().real))
        timer:reset()

        if pool:haserror() then
            pool:synchronize()
        end
        -- print(("stage4: %.6f"):format(timer:time().real))
        enqueue()
        return {input = input, target=target}
    end
    return loop
end

return M.DataLoader
