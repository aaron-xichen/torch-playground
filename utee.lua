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
    local shiftNBits = -torch.ceil(torch.log(maxVal) / torch.log(2))

    -- shift value in place
    for _, v in pairs(xTable) do
        v:mul(2 ^ shiftNBits)
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


-- traverse the whole graph
local traverse
traverse = function(m, orders)
    local layerName = torch.typename(m)
    if m.modules then
        for i=1, #m.modules do
            traverse(m.modules[i], orders)
        end
    elseif layerName ~= 'nn.Sequential' and layerName ~= 'nn.ConcatTable' then
        table.insert(orders, m)
    end
end
M.traverse = traverse

-- generate the topological graph
function M.topologicalOrder(m)
    local orders = {}
    M.traverse(m, orders)
    return orders
end

-- analyze the activation distribution
local analyzeAct
analyzeAct = function(m, input, opt)
    local layerName = torch.typename(m)
    local output
    if m.modules then
        if layerName == 'nn.Sequential' then
            output = input
            for i=1, #m.modules do
                output = analyzeAct(m.modules[i], output, opt)
            end
        elseif layerName == 'nn.ConcatTable' then
            output = {}
            for i=1, #m.modules do
                table.insert(output, analyzeAct(m.modules[i], input, opt))
            end
        else
            assert('Unknown container ' .. layerName)
        end
    else
        output = m:forward(input)
        if (m.weight and m.bias ) or layerName == 'nn.CAddTable' then
            if layerName ~= 'nn.SpatialBatchNormalization' or opt.isQuantizeBN then
                local actShiftBits = -torch.ceil(torch.log(torch.max(torch.abs(output))) / torch.log(2))
                if not m.actShiftBits then
                    m.actShiftBits = actShiftBits
                else
                    m.actShiftBits = math.min(m.actShiftBits, actShiftBits)
                end
            end
        end
    end
    return output
end 
M.analyzeAct = analyzeAct

function M.analyzeActDynamic(model, input, opt)
    local decPosRaw = 0
    local decPosSave = 0
    local maxBitWidth = opt.adderMaxBitWidth

    for i=1, #model do
        if i == 1 then
            ipt = input
        else
            ipt = model:get(i-1).output
        end
        local m = model:get(i)
        local output = m:forward(ipt)
        if m.weight and m.bias then
            local config = opt.bitWidthConfig[i]
            assert(config, ("Bit-width is missing in layer %d"):format(i))

            local actShiftBits = -torch.ceil(torch.log(torch.max(torch.abs(output))) / torch.log(2))
            local weightShiftBits = m.weightShiftBits
            local weightBitWidth, actBitWidth = config[1], config[3]

            m.output:copy(2^-actShiftBits * M.quantization(m.output * 2^actShiftBits, 1, actBitWidth-1))

            if not m.actShiftBits then
                m.actShiftBits = actShiftBits
            else
                m.actShiftBits = math.min(m.actShiftBits, actShiftBits)
            end
        end
    end
end

-- forward with quantization directly
function M.quantizationForwardDirectly(model, input, opt) 
    local ipt
    for i=1, #model do
        if i == 1 then
            ipt = input
        else
            ipt = model:get(i-1).output
        end
        local m = model:get(i)
        local output = m:forward(ipt)
        if m.actShiftBits then
            local config = opt.bitWidthConfig[i]
            assert(config, ("Bit-width is missing in layer %d"):format(i))
            local actBitWidth = config[3]
            if opt.debug then
                local outputTmp1 = output:float()
                print(outputTmp1:sum(), outputTmp1:min(), outputTmp1:max())
            end
            output:copy(2^-m.actShiftBits * M.quantization(output * 2^m.actShiftBits, 1, actBitWidth-1))
            if opt.debug then
                local outputTmp2 = output:float()
                print(outputTmp2:sum(), outputTmp2:min(), outputTmp2:max())
            end
        end
    end
end

-- forward with quantization, handle skip connect
local quantizationForward
quantizationForward = function(m, input, actNBits, debug)
    local layerName = torch.typename(m)
    local output
    if m.modules then
        if layerName == 'nn.Sequential' then
            output = input
            for i=1, #m.modules do
                output = quantizationForward(m.modules[i], output, actNBits)
            end
        elseif layerName == 'nn.ConcatTable' then
            output = {}
            for i=1, #m.modules do
                table.insert(output, quantizationForward(m.modules[i], input, actNBits))
            end
        else
            assert('Unknown container ' .. layerName)
        end
    else
        output = m:forward(input)
        if m.actShiftBits then
            if debug then
                local outputTmp1 = output:float()
                print(outputTmp1:sum(), outputTmp1:min(), outputTmp1:max())
            end
            output:copy(2^-m.actShiftBits * M.quantization(output * 2^m.actShiftBits, 1, actNBits-1))
            if debug then
                local outputTmp2 = output:float()
                print(outputTmp2:sum(), outputTmp2:min(), outputTmp2:max())
            end
        end
    end
    return output
end 
M.quantizationForward = quantizationForward

function M.convertBias(rootPath, meanfilePath, mode)
    assert(rootPath, 'Please specify rootPath')
    assert(meanfilePath, 'Please specify meanfilePath')

    local loadcaffe = require 'loadcaffe'
    require 'nn'
    mode = mode or 'cudnn'
    if mode == 'cudnn' then
        require 'cudnn'
    end

    deployPath = rootPath .. '/deploy.prototxt'
    weightsPath = rootPath .. '/weights.caffemodel'
    local savePath
    if mode == 'cudnn' then
        savePath = rootPath .. '/model.t7'
    else
        savePath = rootPath .. '/modelCPU.t7'
    end

    print("Loading model from " .. rootPath)
    model = loadcaffe.load(deployPath, weightsPath, mode)

    model:apply(
        function(m)
            if m.setMode then m:setMode(1, 1, 1) end
        end
    )

    -- change BGR to RGB
    weight = model:get(1).weight
    tmp = weight[{{}, {1}, {}, {}}]:clone()
    weight[{{}, {1}, {}, {}}] = weight[{{}, {3}, {}, {}}]:clone()
    weight[{{}, {3}, {}, {}}] = tmp

    meanVal = torch.Tensor(torch.load(meanfilePath).mean)
    meanVal = - meanVal:reshape(3, 1, 1):expand(3, 224, 224)
    if mode == 'cudnn' then
        meanVal = meanVal:cuda()
    end
    out = model:get(1):forward(meanVal)
    delta = out[{{}, 100, 100}]
    model:get(1).bias:add(delta)

    print("Saving new model to " .. savePath)
    torch.save(savePath, model)
end

function M.loadTxt(filePath)
    assert(paths.filep(filePath), "File not found")
    print(("Loading from %s"):format(filePath))
    d = {}
    for line in assert(io.open(filePath)):lines() do
        fields = stringx.split(line)
        k = tonumber(fields[1])
        v = {}
        for i=2, #fields do
            table.insert(v, tonumber(fields[i]))
        end
        d[k] = v
    end
    return d
end

function M.saveTxt(filePath, d)
    print(("Saving to %s"):format(filePath))
    writer = io.open(filePath, 'w')
    for k, v in pairs(d) do
        local vStr = table.concat(v, " ")
        local line = tostring(k) .. " " .. vStr .. "\n"
        writer:write(line)
    end
    writer:close()
end

return M

