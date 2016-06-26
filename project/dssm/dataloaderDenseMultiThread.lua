local t = require 'threads'
local M = {}
local DataLoader = torch.class('dssm.DataLoader', M)
t.serialization('threads.sharedserialize')

function DataLoader.create(qFilePath, dFilePath, batchSize, nSamples)
    loader = M.DataLoader(qFilePath, dFilePath, batchSize, nSamples)
    return loader
end

function DataLoader:__init(qFilePath, dFilePath, batchSize, nSamples, nFeature)
    self.batchSize = batchSize
    self.nSamples = nSamples
    self.nFeature = nFeature
    self.nBatch = math.floor(nSamples / batchSize)
    self.qFilePath = qFilePath
    self.dFilePath = dFilePath
    self.pool = t.Threads(3,
        function(threadid)
            require 'pl'
            require 'cudnn'
            print('starting a new thread ' .. threadid)
        end
    )
end 

function DataLoader:size()
    return self.nBatch
end

function DataLoader:nextBatch()
    local pool = self.pool
    local curBatch = 0
    --local inputGPU = torch.CudaTensor(2*self.batchSize, self.nFeature)
    local inputGPU = nil
    local targetGPU = torch.CudaTensor(self.batchSize):fill(1)
    local qOffset = 0
    local dOffset = 0

    local function enqueue()
        while curBatch < self.nBatch and pool:acceptsjob() do
            curBatch = curBatch + 1
            pool:addjob(
                function(qFilePath, dFilePath, qOffset, dOffset, batchSize, nFeature) 
                    local qFileStream = assert(io.open(qFilePath))
                    local dFileStream = assert(io.open(dFilePath))
                    qFileStream:seek('set', qOffset)
                    dFileStream:seek('set', dOffset)
                    -- local input = torch.Tensor(2 * batchSize, nFeature) 
                    -- local input = torch.Tensor(2 * batchSize, 500, 2):fill(0)
                    local inputGPU = torch.CudaTensor(2 * batchSize, nFeature):fill(0)
                    local curLine = 0
                    while true do
                        qline = qFileStream:lines()()
                        dline = dFileStream:lines()()
                        if #qline > 0 and #dline > 0 then
                            curLine = curLine + 1
                            qfields = stringx.split(qline)
                            dfields = stringx.split(dline)
                            for i, pair in ipairs(qfields) do
                                assert(i <= 500, 'out of range')
                                spl = stringx.split(pair, ':')
                                --assert(#spl == 2, 'number not match')
                                key = spl[1]
                                val = spl[2]
                                --input[curLine][i][1] = key + 1
                                --input[curLine][i][2] = val
                                inputGPU[curLine][key + 1] = val
                            end
                            for i, pair in ipairs(dfields) do
                                spl = stringx.split(pair, ':')
                                --assert(#spl == 2, 'number not match')
                                key = spl[1]
                                val = spl[2]
                                --input[curLine + batchSize][i][1] = key + 1
                                --input[curLine + batchSize][i][2] = val

                                inputGPU[curLine + batchSize][key + 1] = val
                            end
                            if curLine >= batchSize then break end
                        end
                    end
                    collectgarbage()
                    return inputGPU, qFileStream:seek(), dFileStream:seek()
                end,
                function(_inputGPU_, _qOffset_, _dOffset_)
                    --input = _input_
                    qOffset = _qOffset_
                    dOffset = _dOffset_
                    --inputGPU:resize(_input_:size()):copy(_input_)
                    inputGPU =_inputGPU_
                end,
                self.qFilePath,
                self.dFilePath,
                qOffset,
                dOffset,
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
        return {input = inputGPU, target=targetGPU}
    end
    return loop
end

return M.DataLoader
