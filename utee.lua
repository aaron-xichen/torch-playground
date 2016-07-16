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
        str = str .. ', ' .. (k .. ': %3.3f'):format(v)
    end
    return str
end

function M.fixedPoint(x, nInt, nFrac)
    local M = 2 ^ (nInt + nFrac) - 1
    local sign = torch.sign(x)
    local floor = torch.floor(torch.abs(x) * 2 ^ nFrac + 0.5)
    local min = torch.cmin(floor, (M - 1) / 2.0)
    local raw = torch.cmul(min, sign)
    return raw 
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

function M.maxShiftNBitsTable(xTable)
    local maxVal = - math.huge
    for _, v in pairs(xTable) do
        maxVal = torch.max(torch.abs(v)) > maxVal and torch.max(torch.abs(v)) or maxVal
    end
    local shiftNBits = torch.ceil(torch.log(maxVal) / torch.log(2))
    
    -- shift value in place
    for _, v in pairs(xTable) do
        v:mul(2 ^ -shiftNBits)
    end
    return shiftNBits
end

function M.overflowRateTable(xTable, nInt, nFrac)
    local M = 2 ^ (nInt + nFrac) - 1
    local delta = 2 ^ -nFrac

    local nCounts, nElements = 0.0, 0.0
    for _, v in pairs(xTable) do
        local floor = torch.floor(torch.abs(v) / delta + 0.5)
        local mask = torch.gt(floor, (M - 1) / 2.0)
        nCounts = nCounts + torch.sum(mask)
        nElements = nElements + v:nElement()
    end
    return nCounts / nElements
end

function M.overflowRate(x, nInt, nFrac)
    local M = 2 ^ (nInt + nFrac) - 1
    local delta = 2 ^ -nFrac
    local floor = torch.floor(torch.abs(x) / delta + 0.5)

    local mask = torch.gt(floor, (M - 1) / 2.0)
    local total = torch.sum(mask)
    return total / x:nElement()
end

function M.copyTo(source, target)
    assert(#source == #target, 'Size does not match')
    for i=1, #source do
        if target:get(i).weight then
            target:get(i).weight:copy(source:get(i).weight)
        end
        if target:get(i).bias then
            target:get(i).bias:copy(source:get(i).bias)
        end
    end
end

function M.substitute(source)
    local layerName = torch.typename(source)
    assert(layerName == 'nn.SpatialConvolution', ('Layer not support %s'):format(layerName))
    local nInputPlane = source.nInputPlane
    local nOutputPlane = source.nOutputPlane
    local kW = source.kW
    local kH = source.kH
    local dW = source.dW
    local dH = source.dH
    local padW = source.padW
    local padH = source.padH
    target = nn.SpatialConvolutionFixedPoint(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    target.weight:copy(source.weight)
    target.bias:copy(source.bias)
    return target
end

return M

