local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('facePair.DataLoader', M)

function DataLoader.create(opt)
    local valDataset = datasets.create(opt, 'val')
    return nil, M.DataLoader(valDataset, opt, 'val')
end

function DataLoader:__init(dataset, opt, split)
    local function init()
        require('datasets/' .. opt.dataset)
    end
    local function main(idx)
        _G.dataset = dataset
        _G.preprocess = dataset:preprocess()
        return dataset:size()
    end

    local threads, sizes = Threads(opt.nThreads, init, main)
    self.threads = threads
    self.__size = sizes[1][1]
    self.batchSize = opt.batchSize
    self.crop = opt.crop
end

function DataLoader:size()
    return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
    local threads = self.threads
    local size, batchSize = self.__size, self.batchSize
    local crop = self.crop
    torch.manualSeed(11)
    local perm = torch.randperm(size)
    local idx, sample = 1, nil
    local function enqueue()
        while idx <= size and threads:acceptsjob() do
            local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
            threads:addjob(
                function(indices)
                    local sz = indices:size(1)
                    local batch1, batch2, target = {}, {}, {} 
                    for i, idx in ipairs(indices:totable()) do
                        local sample = _G.dataset:get(idx)
                        local input1 = _G.preprocess(sample.input[1])
                        local input2 = _G.preprocess(sample.input[2])
                        
                        --print(input1:size(), input2:size())
                        table.insert(batch1, input1)
                        table.insert(batch2, input2)
                        table.insert(target, sample.target)
                        --[[
                        -- first sample in each mini-batch
                        if not batch1 or not batch2 then
                            imageSize = sample.input[1]:size():totable()
                            batch1 = torch.FloatTensor(sz, crop, table.unpack(imageSize))
                            batch2 = torch.FloatTensor(sz, crop, table.unpack(imageSize))
                        end

                        batch1[i]:copy(input1)
                        batch2[i]:copy(input2)
                        ]]--
                        target[i] = sample.target
                    end
                    collectgarbage()
                    return {
                        input1 = torch.cat(batch1, 1):reshape(sz*crop, 3, 224, 224),
                        input2 = torch.cat(batch2, 1):reshape(sz*crop, 3, 224, 224),
                        target = torch.IntTensor(target),
                        --input1 = torch.reshape(batch1, sz*crop, imageSize[1], imageSize[2], imageSize[3]),
                        --input2 = torch.reshape(batch2, sz*crop, imageSize[1], imageSize[2], imageSize[3]),
                        --target = target,
                    }
                end,
                function(_sample_)
                    sample = _sample_
                end,
                indices
            )
            idx = idx + batchSize
        end
    end

    local n = 0
    local function loop()
        enqueue()
        if not threads:hasjob() then
            return nil
        end
        threads:dojob()
        if threads:haserror() then
            threads:synchronize()
        end
        enqueue()
        n = n + 1
        return n, sample
    end
    return loop
end

function DataLoader:reset()
    self.threads:synchronize()
end

return M.DataLoader
