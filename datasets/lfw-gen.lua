require 'image';
local M = {}

-- only process test data
function M.exec(opt, cacheFilePath)
    local valListPath = opt.valListPath
    local imgRoot = opt.data

    valInput = {}
    valTarget = {}

    print(('.Generating data from %s'):format(valListPath))
    local i = 0
    for line in assert(io.open(valListPath)):lines() do
        if i ~= 0 then
            fields = stringx.split(line)
            local label, img1Path, img2Path = -1, nil, nil
            assert(#fields == 3 or #fields == 4, ("Parse error %d"):format(#fields))
            if #fields == 3 then
                label = 1
                local basename1 = fields[1] .. '_' .. ("%04d"):format(fields[2]) .. '.jpg'
                local basename2 = fields[1] .. '_' .. ("%04d"):format(fields[3]) .. '.jpg'

                img1Path = paths.concat(imgRoot, fields[1], basename1)
                img2Path = paths.concat(imgRoot, fields[1], basename2)
            else
                label = 0
                local basename1 = fields[1] .. '_' .. ("%04d"):format(fields[2]) .. '.jpg'
                local basename2 = fields[3] .. '_' .. ("%04d"):format(fields[4]) .. '.jpg'

                img1Path = paths.concat(imgRoot, fields[1], basename1)
                img2Path = paths.concat(imgRoot, fields[3], basename2)
            end
            table.insert(valInput, image.load(img1Path, 3, 'byte'))
            table.insert(valInput, image.load(img2Path, 3, 'byte'))
            table.insert(valTarget, label)
        end
        i = i + 1
    end

    print('.Converting table to compact tensor')
    local shape = torch.totable(valInput[1]:size())
    valInputTensor = torch.cat(valInput, 1):view(-1, 2, unpack(shape)):contiguous()
    valTargetTensor = torch.ByteTensor(valTarget):contiguous()

    valData = {
        data = valInputTensor,
        labels = valTargetTensor
    }

    print(".Saving LFW dataset to " .. cacheFilePath)
    torch.save(
        cacheFilePath, 
        {
            train = {},
            val = valData,
        }
    )
end

return M
