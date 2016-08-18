require 'nn';
require 'cutorch';
require 'loadcaffe';
require 'struct'

torch.setdefaulttensortype('torch.FloatTensor')

function quantization(x, nInt, nFrac)
    local M = 2 ^ (nInt + nFrac) - 1
    local delta = 2 ^ -nFrac
    local sign = torch.sign(x)
    local floor = torch.floor(torch.abs(x) / delta + 0.5)
    local min = torch.cmin(floor, (M - 1) / 2.0)
    local raw = torch.mul(torch.cmul(min, sign), delta)
    return raw
end

function fixedPoint(x, nInt, nFrac)
    local M = 2 ^ (nInt + nFrac) - 1
    local sign = torch.sign(x)
    local floor = torch.floor(torch.abs(x) * 2 ^ nFrac + 0.5)
    local min = torch.cmin(floor, (M - 1) / 2.0)
    local raw = torch.cmul(min, sign)
    return raw 
end

function substitute(source)
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

imgs = torch.load('input.t7'):int()[1]:view(1, 3, 224, 224)
print('image range: ', torch.min(imgs), torch.max(imgs))

-- weight, bias, output, biasAlign, winShift, docPosSave, docPosRaw
shiftTable = torch.load('meta.t7')

modelcpu = torch.load('/home/chenxi/modelzoo/vgg16/modelCPU.t7')

modelcpu:evaluate()

cpuFixed = modelcpu:clone()
for i=1,#modelcpu do
    if modelcpu:get(i).weight then
        local weight = modelcpu:get(i).weight:clone()

        local weight1 = quantization(2^shiftTable[i][1] * weight, 1, 7) * 2^-shiftTable[i][1]
        modelcpu:get(i).weight:copy(weight1)

        local weight2 = fixedPoint(2^shiftTable[i][1] * weight, 1, 7)
        cpuFixed:get(i).weight:copy(weight2)

    end

    if modelcpu:get(i).bias then
        local bias = modelcpu:get(i).bias:clone()

        local bias1 = quantization(2^shiftTable[i][2] * bias, 1, 7) * 2^-shiftTable[i][2]
        modelcpu:get(i).bias:copy(bias1)

        -- rounding version
        --[[
        local nbit = math.max(math.min(shiftTable[i][4], 0) + 7, 0)
        local bias2 = torch.round((fixedPoint(2^shiftTable[i][2] * bias, 1, 7) * 2 ^shiftTable[i][4]))
        if shiftTable[i][4] < 0 then
        bias2:clamp(-2^nbit+1, 2^nbit-1)
    end
        ]]--

        -- clip version
        local bias2 = fixedPoint(2^shiftTable[i][2] * bias, 1, 7) * 2 ^shiftTable[i][4]

        cpuFixed:get(i).bias:copy(bias2)
    end

    if modelcpu:get(i).inplace then
        modelcpu:get(i).inplace = false
        cpuFixed:get(i).inplace = false
    end

    local layerName = torch.typename(modelcpu:get(i))
    if layerName == 'nn.SoftMax' then
        modelcpu:remove(i)
        cpuFixed:remove(i)
    end
end

-- cpu float net
cpuFloat = modelcpu:clone()
for i=1, #cpuFloat do
    cpuFloat:get(i):type('torch.FloatTensor')
end

-- cpu fixed point net
print('Substituting SpatialConvolution with SpationConvolutionFixedPoint')
for i=1,#cpuFixed do
    local layerName = torch.typename(cpuFixed:get(i))
    if layerName == 'nn.SpatialConvolution' then
        local tmp = cpuFixed:get(i):clone()
        cpuFixed:remove(i)
        cpuFixed:insert(substitute(tmp), i)
    end
    cpuFixed:get(i):type('torch.IntTensor')
end

print(modelcpu)

local rootFolderName = 'golden'
print("Saving params")
if paths.dirp(rootFolderName) then
    print("Detect old " .. rootFolderName .. ", delete it")
    assert(paths.rmall(rootFolderName, 'yes'), 'Delete ' .. rootFolderName .. ' fail')
end
print("Creating " .. rootFolderName)
assert(paths.mkdir(rootFolderName), 'Create ' .. rootFolderName .. ' fail')
for i=1, #cpuFixed do
    if cpuFixed:get(i).weight then
        local layerName = torch.typename(cpuFixed:get(i))
        print(layerName)

        local subFolderName = paths.concat(rootFolderName,  i .. layerName)
        if not paths.dirp(subFolderName) then
            print("Creating " .. subFolderName)
            assert(paths.mkdir(subFolderName), 'Create ' .. subFolderName .. ' fail')
        end

        -- save weight
        local weightFixed
        if layerName == 'nn.Linear' then
            weightFixed = cpuFixed:get(i).weight:transpose(1, 2):contiguous():view(-1)
        else
            weightFixed = cpuFixed:get(i).weight:view(-1)
        end
        local biasFixed = cpuFixed:get(i).bias:view(-1)


        local weightPath = paths.concat(subFolderName, 'weight.bin')
        local biasPath = paths.concat(subFolderName, 'bias.bin')
        local weightWriter = assert(io.open(weightPath, 'wb'))
        local biasWriter = assert(io.open(biasPath, 'wb'))
        for i=1, weightFixed:nElement() do
            weightWriter:write(struct.pack('<i1', weightFixed[i]))
        end
        for i=1, biasFixed:nElement() do
            biasWriter:write(struct.pack('<i1', biasFixed[i]))
        end
        weightWriter:close()
        biasWriter:close()
    end
end

-- write imgs
print("Saving image")
local imgPath = paths.concat(rootFolderName, 'img.bin')
local imgWriter = assert(io.open(imgPath, 'wb'))
for i=1, imgs:nElement() do
    imgWriter:write(struct.pack('<I1', imgs:view(-1)[i]))
end
imgWriter:close()


-- forward
print("Saving output")
for i=1, #modelcpu do

    -- forward
    if i == 1 then
        cpuFloat:get(i):forward(imgs:float())
        cpuFixed:get(i):forward(imgs:int())
    else
        cpuFloat:get(i):forward(cpuFloat:get(i-1).output)
        cpuFixed:get(i):forward(cpuFixed:get(i-1).output)
    end

    -- io context
    local layerName = torch.typename(cpuFixed:get(i))
    print(layerName)
    local subFolderName = paths.concat(rootFolderName,  i .. layerName)
    if not paths.dirp(subFolderName) then
        print("Creating " .. subFolderName)
        assert(paths.mkdir(subFolderName), 'Create ' .. subFolderName .. ' fail')
    end

    local cpuFloatOutput = cpuFloat:get(i).output
    local cpuFixedOutput = cpuFixed:get(i).output
    if shiftTable[i] then
        local cpuFixedOutputTmp1 = cpuFixedOutput:float() * 2^shiftTable[i][7]

        -- save actPre value
        local actPrePath = paths.concat(subFolderName, 'actPre.bin')
        local actPreWriter = assert(io.open(actPrePath, 'wb'))
        for i=1, cpuFixedOutput:nElement() do
            actPreWriter:write(struct.pack('<i4', cpuFixedOutput:view(-1)[i]))
        end
        actPreWriter:close()

        print('flt: ', cpuFloatOutput:sum(), cpuFloatOutput:min(), cpuFloatOutput:max())
        print('fix: ', cpuFixedOutputTmp1:sum(), cpuFixedOutputTmp1:min(), cpuFixedOutputTmp1:max())

        cpuFloatOutput:copy(quantization(2^shiftTable[i][3] * cpuFloatOutput, 1, 7) * 2 ^ -shiftTable[i][3])

        local shiftLeft = bit.lshift(0x7f, shiftTable[i][5])
        local overflow = bit.lshift(0x80, shiftTable[i][5]) 
        local roundBit = bit.lshift(0x1, shiftTable[i][5] - 1)
        local sign = torch.sign(cpuFixedOutput)
        cpuFixedOutput:abs():apply(
            function(x)
                if bit.band(x, overflow) ~= 0 then  -- overflow, return max
                    return 127
                elseif bit.band(x, roundBit) ~= 0 then -- ceil
                    return math.min(bit.rshift(bit.band(x, shiftLeft), shiftTable[i][5]) + 1, 127)
                else -- floor
                    return bit.rshift(bit.band(x, shiftLeft), shiftTable[i][5])
                end
            end
        )
        cpuFixedOutput:cmul(sign)

        local cpuFixedOutputTmp2 = cpuFixedOutput:float() * 2^shiftTable[i][6]

        print('flt: ', cpuFloatOutput:sum(), cpuFloatOutput:min(), cpuFloatOutput:max())
        print('fix: ', cpuFixedOutputTmp2:sum(), cpuFixedOutputTmp2:min(), cpuFixedOutputTmp2:max())
    end

    -- save actPost value
    local actPostPath = paths.concat(subFolderName, 'actPost.bin')
    local actPostWriter = assert(io.open(actPostPath, 'wb'))
    for i=1, cpuFixedOutput:nElement() do
        actPostWriter:write(struct.pack('<i1', cpuFixedOutput:view(-1)[i]))
    end
    actPostWriter:close()

end