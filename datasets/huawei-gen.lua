require 'image';
local M = {}

function M.exec(opt, cacheFilePath)
    local trainListPath = opt.trainListPath
    local valListPath = opt.valListPath
    local imgRoot = opt.imgRoot

    trainInput = {}
    trainTarget = {}
    valInput = {}
    valTarget = {}

    print(('.Generating data from %s'):format(trainListPath))
    local i = 0
    for line in assert(io.open(trainListPath)):lines() do
        fields = stringx.split(line)
        imgPath = fields[1]
        table.remove(fields, 1)
        if torch.sum(torch.Tensor(fields)) > 0 then
            table.insert(trainTarget, fields)

            img = image.load(paths.concat(imgRoot, imgPath), 3, 'byte')
            table.insert(trainInput, img)
            i = i + 1
        end
    end

    i = 0
    print(('.Generating data from %s'):format(valListPath))
    for line in assert(io.open(valListPath)):lines() do
        fields = stringx.split(line)
        imgPath = fields[1]
        table.remove(fields, 1)
        if torch.sum(torch.Tensor(fields)) > 0 then
            table.insert(valTarget, fields)

            img = image.load(paths.concat(imgRoot, imgPath), 3, 'byte')
            table.insert(valInput, img)
            i = i + 1
        end
    end

    print(".Joining data into a single file")
    
    
    local shape = torch.totable(trainInput[1]:size())
    trainInputTensor = torch.cat(trainInput, 1):view(-1, unpack(shape)):contiguous()
    trainTargetTensor = torch.ByteTensor(trainTarget):contiguous()
    valInputTensor = torch.cat(valInput, 1):view(-1, unpack(shape)):contiguous()
    valTargetTensor = torch.ByteTensor(valTarget):contiguous()

    trainData = {
        data = trainInputTensor,
        labels = trainTargetTensor
    }
    valData = {
        data = valInputTensor,
        labels = valTargetTensor
    }

    print(".Saving Huawei Scene labeling dataset to " .. cacheFilePath)
    torch.save(
        cacheFilePath, 
        {
            train = trainData,
            val = valData,
        }
    )
end

return M
