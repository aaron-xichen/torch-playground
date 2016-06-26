local M = { }

function M.save(model, optimState, epoch, snapshotPath)
    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    local snapshot = {
        model = model,
        optimState = optimState,
        epoch = epoch
    }

    if paths.filep(snapshotPath) then 
        print(("... Removing previous best snapshot %s"):format(snapshotPath))
        os.remove(snapshotPath) 
    end
    print(("... Saving best snapshot to %s"):format(snapshotPath))
    torch.save(snapshotPath, snapshot)
end


function M.convertToString(keys, vals)
    assert(#keys == #vals, 'Size does not match')
    local str = ''
    for i = 1, #keys do
        k = keys[i]
        v = vals[i]
        assert(type(v) == 'number', 'Unknown Type')
        str = str .. (k .. ': %3.3f, '):format(v)
        --[[
        if type(v) == 'number' then
            str = str .. (k .. ': %3.3f '):format(v)
        elseif torch.typename(v) == 'torch.FloatTensor' then
            str = str .. k .. ':'
            for i = 1, v:nElement() do
                str = str .. (' %3.3f'):format(v[i])
            end
            str = str .. ' '
        else
            assert(nil, 'Unknown type')
        end]]--
    end
    return str
end


return M
