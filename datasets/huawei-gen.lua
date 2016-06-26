require 'image';
local M = {}

function join(input)
    inputSize = torch.totable(input[1]:size())
    table.insert(inputSize, 1, -1)
    m = nn.Sequential()
    m:add(nn.JoinTable(1))
    m:add(nn.View(unpack(inputSize)))
    return m:forward(input):byte()
end


function M.exec(opt, cacheFile)
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

            img = image.load(imgRoot .. imgPath, 3, 'byte')
            table.insert(trainInput, img)
            i = i + 1
        end
        if i == 64 then break end
    end

    i = 0
    print(('.Generating data from %s'):format(valListPath))
    for line in assert(io.open(valListPath)):lines() do
        fields = stringx.split(line)
        imgPath = fields[1]
        table.remove(fields, 1)
        if torch.sum(torch.Tensor(fields)) > 0 then
            table.insert(valTarget, fields)

            img = image.load(imgRoot .. imgPath, 3, 'byte')
            table.insert(valInput, img)
            i = i + 1
        end
        if i == 64 then break end
    end

    print(".Joining data into a single file")
    trainInputTensor = join(trainInput):contiguous()
    trainTargetTensor = torch.ByteTensor(trainTarget):contiguous()
    valInputTensor = join(valInput):contiguous()
    valTargetTensor = torch.ByteTensor(valTarget):contiguous()

    trainData = {
        data = trainInputTensor,
        labels = trainTargetTensor
    }
    valData = {
        data = valInputTensor,
        labels = valTargetTensor
    }

    local tmp = trainInputTensor:transpose(1, 2):contiguous():view(3, -1)
    local mean = torch.mean(tmp:float(), 2):squeeze()
    local std = torch.std(tmp:float(), 2):squeeze()
    local meanstd = {
        mean = torch.totable(mean),
        std = torch.totable(std)
    }
    print(".Saving Huawei Scene labeling dataset to " .. cacheFile)
    print('.Mean and std', meanstd)
    torch.save(
        cacheFile, 
        {
            train = trainData,
            val = valData,
            meanstd = meanstd
        }
    )
end

return M
