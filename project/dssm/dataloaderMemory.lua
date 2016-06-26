require 'pl';
local M = {}
local DataLoader = torch.class('dssm.DataLoader', M)


function DataLoader.create(dataFilePath, batchSize, nFeature)
    loader = M.DataLoader(dataFilePath, batchSize, nFeature)
    return loader
end
            
function DataLoader:__init(dataFilePath, batchSize, nFeature)
    self.batchSize = batchSize
    self.nFeature = nFeature
    
    -- load the t7 file
    print("Loading data from " .. dataFilePath)
    local data = torch.load(dataFilePath)
    self.query = data['query']
    self.doc = data['doc']
    
    assert(self.query:size()[1] == self.doc:size()[1], 'Size does not match')
    print(("Loaded query: %d, doc: %d"):format(self.query:size()[1], self.doc:size()[1]))
    
    self.nSamples = self.query:size()[1]
    self.nBatch = math.floor(self.nSamples / self.batchSize)
end 

function DataLoader:size()
   return self.nBatch
end

function DataLoader:nextBatch()
    local curBatch = 0
    local perm = torch.randperm(self.nSamples)
    local data = torch.Tensor(2 * self.batchSize, self.nFeature)
    local label = torch.Tensor(self.batchSize):fill(1)
    
    function loop()
        curBatch = curBatch + 1
        if curBatch <= self.nBatch then
            data:fill(0)
            local beginIdx = self.batchSize * (curBatch - 1) + 1
            for j=1, self.batchSize do
                local idx = perm[beginIdx + j - 1]
                
                for i=1, self.query:size()[2] do
                    key = self.query[idx][i][1]
                    val = self.query[idx][i][2]
                    if key == 0 then break end
                    data[j][key] = val
                end
                
                for i=1, self.doc:size()[2] do
                    key = self.doc[idx][i][1]
                    val = self.doc[idx][i][2]
                    if key == 0 then break end
                    data[j + self.batchSize][key] = val
                end
            end
            -- collectgarbage()
            return {input=data, target=label}
        else
            return nil
        end
   end
   return loop
end

return M.DataLoader
