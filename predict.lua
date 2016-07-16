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
mean = {104, 117, 124}
imgs:add(-torch.IntTensor(mean):view(1, 3, 1, 1):expandAs(imgs))
print(torch.max(imgs))
print(torch.min(imgs))

-- weight, bias, output, biasAlign, winShift, docPosSave, docPosRaw
shiftTable = {
    [1] = {0, -2, -10, 2, 10, 3, -7},
    [3] = {1, -1, -12, -1, 10, 5, -5},
    [6] = {1, 1, -13, -5, 9, 6, -3},
    [8] = {1, 0, -14, -5, 9, 7, -2},
    [11] = {0, 1, -15, -8, 8, 8, 0},
    [13] = {1, 1, -14, -8, 7, 7, 0},
    [15] = {1, 0, -15, -6, 9, 8, -1},
    [18] = {1, 1, -14, -8, 7, 7, 0},
    [20] = {1, 2, -13, -8, 7, 6, -1},
    [22] = {1, 1, -12, -6, 7, 5, -2},
    [25] = {2, 0, -12, -3, 9, 5, -4},
    [27] = {2, 0, -11, -3, 8, 4, -4},
    [29] = {1, -4, -10, 1, 7, 3, -4},
    [33] = {5, 0, -7, 2, 9, 0, -9},
    [36] = {4, -1, -5, 5, 9, -2, -11},
    [39] = {4, 0, -6, 6, 12, -1, -13}
}


modelcpu = loadcaffe.load('/home/chenxi/modelzoo/vgg16/deploy.prototxt', '/home/chenxi/modelzoo/vgg16/weights.caffemodel', 'nn')
modelgpu = loadcaffe.load('/home/chenxi/modelzoo/vgg16/deploy.prototxt', '/home/chenxi/modelzoo/vgg16/weights.caffemodel', 'cudnn')

modelcpu:evaluate()
modelgpu:evaluate()

cpuFixed = modelcpu:clone()
for i=1,#modelcpu do
    if modelcpu:get(i).weight then
        local weight = modelcpu:get(i).weight:clone()

        local weight1 = quantization(2^shiftTable[i][1] * weight, 1, 7) * 2^-shiftTable[i][1]
        modelcpu:get(i).weight:copy(weight1)
        modelgpu:get(i).weight:copy(weight1)

        local weight2 = fixedPoint(2^shiftTable[i][1] * weight, 1, 7)
        cpuFixed:get(i).weight:copy(weight2)

    end
    if modelcpu:get(i).bias then
        local bias = modelcpu:get(i).bias:clone()

        local bias1 = quantization(2^shiftTable[i][2] * bias, 1, 7) * 2^-shiftTable[i][2]
        modelcpu:get(i).bias:copy(bias1)
        modelgpu:get(i).bias:copy(bias1)

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
        modelgpu:get(i).inplace = false
        cpuFixed:get(i).inplace = false
    end

    local layerName = torch.typename(modelcpu:get(i))
    if layerName == 'nn.SoftMax' then
        modelcpu:remove(i)
        modelgpu:remove(i)
        cpuFixed:remove(i)
    end
end

-- cpu float net
cpuFloat = modelcpu:clone()
for i=1, #cpuFloat do
    cpuFloat:get(i):type('torch.FloatTensor')
end

-- cpu double net
cpuDouble = modelcpu:clone()
for i=1, #cpuDouble do
    cpuDouble:get(i):type('torch.DoubleTensor')
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
--[[
-- write params
print("Saving params")
--outFloatParam = assert(io.open('floatParam.bin', 'wb'))
outInt32Param = assert(io.open('goldenParam.bin', 'wb'))
local sum1, sum2, sum3, sum4 = 0, 0, 0, 0
for i=1, #cpuFixed do
if cpuFixed:get(i).weight then
-- save weight
local weightFixed
local biasFixed = cpuFixed:get(i).bias:view(-1)
local layerName = torch.typename(cpuFixed:get(i))
print(layerName)
if layerName == 'nn.Linear' then
-- weightFloat = cpuFloat:get(i).weight:transpose(1, 2):contiguous():view(-1)
weightFixed = cpuFixed:get(i).weight:transpose(1, 2):contiguous():view(-1)
sum3 = sum3 + weightFixed:nElement()
sum4 = sum4 + biasFixed:nElement()
else
--weightFloat = cpuFloat:get(i).weight:view(-1)
weightFixed = cpuFixed:get(i).weight:view(-1)
sum1 = sum1 + weightFixed:nElement()
sum2 = sum2 + biasFixed:nElement()
end
for i=1, weightFixed:nElement() do
-- outFloatParam:write(struct.pack('<f', weightFloat[i]))
outInt32Param:write(struct.pack('<i1', weightFixed[i]))
end
for i=1, biasFixed:nElement() do
-- outFloatParam:write(struct.pack('<f', biasFloat[i]))
outInt32Param:write(struct.pack('<i1', biasFixed[i]))
end
print(sum1, sum2, sum3, sum4)
end
end
outFloatParam:close()
--outInt32Param:close()

-- write imgs
print("Saving image")
outImgs = assert(io.open('imgs.bin', 'wb'))
for i=1, imgs:nElement() do
outImgs:write(struct.pack('<I1', imgs:view(-1)[i]))
end
outImgs:close()
]]--

-- forward
print("Saving output")
outFloatAct = assert(io.open('goldenAct.bin', 'wb'))
outInt32Act = assert(io.open('silverAct.bin', 'wb'))
local actShift = 0
local total = 0
for i=1, #modelcpu do
    if i == 1 then
        cpuFloat:get(i):forward(imgs:float())
        cpuDouble:get(i):forward(imgs:double())
        cpuFixed:get(i):forward(imgs:int())
        modelgpu:get(i):forward(imgs:cuda())
    else
        cpuFloat:get(i):forward(cpuFloat:get(i-1).output)
        cpuDouble:get(i):forward(cpuDouble:get(i-1).output)
        cpuFixed:get(i):forward(cpuFixed:get(i-1).output)
        modelgpu:get(i):forward(modelgpu:get(i-1).output)
    end

    -- quantization
    local floatAct, int32Act

    local layerName = torch.typename(modelcpu:get(i))
    print(layerName)
    if shiftTable[i] then
        local cpuFloatOutput = cpuFloat:get(i).output
        local cpuDoubleOutput = cpuDouble:get(i).output
        local cpuFixedOutput = cpuFixed:get(i).output
        local cpuFixedOutputTmp1 = cpuFixedOutput:float() * 2^shiftTable[i][7]
        local gpuOutput = modelgpu:get(i).output

        print('flt: ', cpuFloatOutput:sum(), cpuFloatOutput:min(), cpuFloatOutput:max())
        print('dbl: ', cpuDoubleOutput:float():sum(), cpuDoubleOutput:float():min(), cpuDoubleOutput:float():max())
        print('fix: ', cpuFixedOutputTmp1:sum(), cpuFixedOutputTmp1:min(), cpuFixedOutputTmp1:max())
        print('gpu: ', gpuOutput:float():sum(), gpuOutput:float():min(), gpuOutput:float():max())

        floatAct = fixedPoint(2^shiftTable[i][3] * cpuFloatOutput, 1, 7):int()

        cpuFloatOutput:copy(quantization(2^shiftTable[i][3] * cpuFloatOutput, 1, 7) * 2 ^ -shiftTable[i][3])
        cpuDoubleOutput:copy(quantization(2^shiftTable[i][3] * cpuDoubleOutput, 1, 7) * 2 ^ -shiftTable[i][3])
        gpuOutput:copy(quantization(2^shiftTable[i][3] * gpuOutput, 1, 7) * 2 ^ -shiftTable[i][3])

        local shiftLeft = bit.lshift(0x7f, shiftTable[i][5])
        local overflow = bit.lshift(0x80, shiftTable[i][5]) 
        local roundBit = bit.lshift(0x1, shiftTable[i][5] - 1)
        local sign = torch.sign(cpuFixedOutput)
        cpuFixedOutput:abs():apply(
            function(x)
                -- overflow, return max
                if bit.band(x, overflow) ~= 0 then 
                    return 127
                    -- ceil
                elseif bit.band(x, roundBit) ~= 0 then
                    return math.min(bit.rshift(bit.band(x, shiftLeft), shiftTable[i][5]) + 1, 127)
                    -- floor
                else
                    return bit.rshift(bit.band(x, shiftLeft), shiftTable[i][5])
                end
            end
        )
        cpuFixedOutput:cmul(sign)

        int32Act = cpuFixedOutput:int()

        local cpuFixedOutputTmp2 = cpuFixedOutput:float() * 2^shiftTable[i][6]

        print('flt: ', cpuFloatOutput:sum(), cpuFloatOutput:min(), cpuFloatOutput:max())
        print('dbl: ', cpuDoubleOutput:float():sum(), cpuDoubleOutput:float():min(), cpuDoubleOutput:float():max())
        print('fix: ', cpuFixedOutputTmp2:sum(), cpuFixedOutputTmp2:min(), cpuFixedOutputTmp2:max())
        print('gpu: ', gpuOutput:float():sum(), gpuOutput:float():min(), gpuOutput:float():max())
        actShift = shiftTable[i][3]
    else
        floatAct = fixedPoint( 2^actShift * cpuFloat:get(i).output, 1, 7):int()
        int32Act = cpuFixed:get(i).output:int()
    end


    if layerName ~= 'nn.View' and layerName ~= 'nn.Dropout' then
        print('compare:', floatAct:sum(), int32Act:sum())
        print(floatAct:size())
        total = total + floatAct:nElement()
        for i=1, floatAct:nElement() do
            outFloatAct:write(struct.pack('<i1', floatAct:view(-1)[i]))
            outInt32Act:write(struct.pack('<i1', int32Act:view(-1)[i]))
        end
    end
end
print(total)
outFloatAct:close()
outInt32Act:close()