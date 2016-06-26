require 'pl';
local M = {}
local DataLoader = torch.class('dssm.DataLoader', M)


function DataLoader.create(qFilePath, dFilePath, batchSize, nSamples)
    loader = M.DataLoader(qFilePath, dFilePath, batchSize, nSamples)
    return loader
end

function DataLoader:__init(qFilePath, dFilePath, batchSize, nSamples, nFeature)
    self.batchSize = batchSize
    self.nSamples = nSamples
    self.nFeature = nFeature
    self.nBatch = math.floor(nSamples / batchSize)
    self.qFileStream = assert(io.open(qFilePath))
    self.dFileStream = assert(io.open(dFilePath))
end 

function DataLoader:size()
    return self.nBatch
end

function DataLoader:nextBatch()
    local curBatch = 0
    local curLine = 0
    local data = torch.Tensor(2 * self.batchSize, 300, 2)
    local label = torch.Tensor(self.batchSize):fill(1)
    function loop()
        curLine = 0
        data:select(3, 1):fill(self.nFeature + 1)
        data:select(3, 2):fill(0)
        curBatch = curBatch + 1
        if curBatch > self.nBatch then 
            self.qFileStream:seek("set")
            self.dFileStream:seek("set")
            curBatch = 0
            return nil 
        end
        while true do
            qline = self.qFileStream:lines()()
            dline = self.dFileStream:lines()()
            if #qline > 0 and #dline > 0 then
                curLine = curLine + 1
                qfields = stringx.split(qline)
                dfields = stringx.split(dline)

                for i, pair in ipairs(qfields) do
                    assert(i <= 300, 'out of range')
                    spl = stringx.split(pair, ':')
                    key = spl[1]
                    val = spl[2]
                    data[curLine][i][1] = key + 1
                    data[curLine][i][2] = val
                end
                for i, pair in ipairs(dfields) do
                    assert(i <= 300, 'out of range')
                    spl = stringx.split(pair, ':')
                    key = spl[1]
                    val = spl[2]
                    data[curLine + self.batchSize][i][1] = key + 1
                    data[curLine + self.batchSize][i][2] = val
                end

                if curLine >= self.batchSize then break end
            end
        end
        collectgarbage()
        return {input=data, target=label}
    end
    return loop
end

return M.DataLoader
