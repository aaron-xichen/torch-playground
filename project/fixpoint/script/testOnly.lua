require 'nn';
require 'cutorch';
require 'loadcaffe';

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

function computeScore(output, target, nCrops)
    if nCrops > 1 then
        output = output:view(output:size(1) / nCrops, nCrops, output:size(2)):sum(2):squeeze(2)
    end
    local batchSize = output:size(1)

    local _ , predictions = output:float():sort(2, true) -- descending

    local correct = predictions:eq(
        target:long():view(batchSize, 1):expandAs(output))

    local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

    local len = math.min(5, correct:size(2))
    local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

    return top1 * 100, top5 * 100
end


----------------
opt = {}
opt.nThreads = 1
opt.batchSize = 50
opt.gen = 'gen'
opt.dataset = 'imagenet'
opt.data = '/work/shadow/'

local DataLoader = require('dataloader')
local _, valLoader = DataLoader.create(opt)

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


modelcpu:evaluate()

torch.manualSeed(11)

cpuFixed = modelcpu:clone()
for i=1,#modelcpu do
    if modelcpu:get(i).weight then
        local weight = modelcpu:get(i).weight:clone()

        local weight2 = fixedPoint(2^shiftTable[i][1] * weight, 1, 7)
        cpuFixed:get(i).weight:copy(weight2)

    end
    if modelcpu:get(i).bias then
        local bias = modelcpu:get(i).bias:clone()

        local bias2 = torch.floor(fixedPoint(2^shiftTable[i][2] * bias, 1, 7) * 2 ^shiftTable[i][4])
        cpuFixed:get(i).bias:copy(bias2)
    end

    if modelcpu:get(i).inplace then
        cpuFixed:get(i).inplace = false
    end

    local layerName = torch.typename(modelcpu:get(i))
    if layerName == 'nn.SoftMax' then
        cpuFixed:remove(i)
    end
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

local size = valLoader:size()
local nCrops = 1
local top1Sum, top5Sum = 0.0, 0.0
local N = 0
local totalTimer = torch.Timer()
local timer = torch.Timer()
for n, sample in valLoader:run() do
    print('data: ', torch.sum(sample.input:int():float()))
    timer:reset()

    -- forward
    for i=1, #cpuFixed do
        if i == 1 then
            cpuFixed:get(i):forward(sample.input:int())
        else
            cpuFixed:get(i):forward(cpuFixed:get(i-1).output)
        end

        -- quantization
        if shiftTable[i] then

            local cpuFixedOutput = cpuFixed:get(i).output
            --local cpuFixedOutputTmp1 = cpuFixedOutput:float() * 2^shiftTable[i][7]
            --print(i)

            --print('fix: ', cpuFixedOutputTmp1:sum(), cpuFixedOutputTmp1:min(), cpuFixedOutputTmp1:max())

            local shiftLeft = bit.lshift(0x7f, shiftTable[i][5])
            local overflow = bit.lshift(0x80, shiftTable[i][5]) 
            local roundBit = bit.lshift(0x1, shiftTable[i][5] - 1)
            local sign = torch.sign(cpuFixedOutput)
            cpuFixedOutput:abs():apply(
                function(x)
                    if bit.band(x, overflow) ~= 0 then 
                        return 127
                    elseif bit.band(x, roundBit) ~= 0 then
                        return math.min(bit.rshift(bit.band(x, shiftLeft), shiftTable[i][5]) + 1, 127)
                    else
                        return bit.rshift(bit.band(x, shiftLeft), shiftTable[i][5])
                    end
                end
            )
            cpuFixedOutput:cmul(sign)

            --local cpuFixedOutputTmp2 = cpuFixedOutput:float() * 2^shiftTable[i][6]

            --print('fix: ', cpuFixedOutputTmp2:sum(), cpuFixedOutputTmp2:min(), cpuFixedOutputTmp2:max())
        end
    end


    local top1, top5 = computeScore(cpuFixed:get(#cpuFixed).output, sample.target, nCrops)
    top1Sum = top1Sum + top1
    top5Sum = top5Sum + top5
    N = N + 1

    print((' | Val [%d/%d] Top1Err: %7.5f, Top5Err: %7.5f, cost: %3.3fs'):format(n, size, top1, top5, timer:time().real))
    
    --valLoader:reset()
    --break
end

print((' * Val Done, Top1Err: %7.3f  Top5Err: %7.3f, cost: %3.3fs'):format(top1Sum / N, top5Sum / N, totalTimer:time().real))