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


function M.quantization(x, nInt, nFrac)
    local M = 2 ^ (nInt + nFrac) - 1
    local delta = 2 ^ -nFrac
    local sign = torch.sign(x)
    local floor = torch.floor(torch.abs(x) / delta + 0.5)
    local min = torch.cmin(floor, (M - 1) / 2.0)
    local raw = torch.mul(torch.cmul(min, sign), delta)
    return raw
end

function M.overflowRate(x, nInt, nFrac)
    local M = 2 ^ (nInt + nFrac) - 1
    local delta = 2 ^ -nFrac
    local sign = torch.sign(x)
    local floor = torch.floor(torch.abs(x) / delta + 0.5)
    
    local mask = torch.gt(floor, (M - 1) / 2.0)
    local total = torch.sum(mask)
    return total / x:nElement()
end

return M
