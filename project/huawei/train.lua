local optim = require 'optim'

local M = {}
local Trainer = torch.class('huawei.Trainer', M)

function Trainer:__init(model, criterion, optimState, opt, trainDataLoader, valDataLoader)
    self.model = model
    self.criterion = criterion
    self.opt = opt

    self.params, self.gradParams = self.model:parameters()
    self.optimStates = {}

    -- for different learning rate
    for i=1, #self.params do
        table.insert(self.optimStates,
            {
                learningRate = 0,
                learningRateDecay = 0.0,
                momentum = opt.momentum,
                nesterov = true,
                dampening = 0.0,
                weightDecay = opt.weightDecay,
            }
        )
    end

    self.trainDataLoader = trainDataLoader
    self.valDataLoader = valDataLoader
end

function Trainer:getBestStat()
    keys = {'bestHitEach', 'bestHitAll', 'bestHitOne', 'bestTop1', 'bestLoss'}
    vals = {-math.huge, -math.huge, -math.huge, -math.huge, math.huge}
    return keys, vals
end

function Trainer:train(epoch)
    assert(#self.params % 2 == 0, "Error")
    local parametricLayers = #self.params / 2
    print("Finetuning last " .. self.opt.last .. " layers")
    for i=1, parametricLayers-self.opt.last do
        self.optimStates[2*i-1].learningRate = self:learningRate(epoch) * self.opt.lrRatio -- for W
        self.optimStates[2*i].learningRate = 2 * self:learningRate(epoch) * self.opt.lrRatio -- for b
    end
    
    for i=parametricLayers-self.opt.last+1, parametricLayers do
        self.optimStates[2*i-1].learningRate = self:learningRate(epoch) -- for W
        self.optimStates[2*i].learningRate = 2 * self:learningRate(epoch) -- for b
    end


    local size = self.trainDataLoader:size()
    local hitAllSum, hitOneSum, lossSum = 0.0, 0.0, 0.0
    local individualSum = nil

    local N = 0

    local lrFormer = self:learningRate(epoch) * self.opt.lrRatio
    local lrLatter = self:learningRate(epoch)
    print(('=> Training epoch: #%d, LR-Former, %.3e, LR-Last: %.3e'):format(epoch, lrFormer, lrLatter))
    self.model:training()

    local timer = torch.Timer()
    for n, sample in self.trainDataLoader:nextBatch() do
        local dataTime = timer:time().real
        timer:reset()

        self:copyInputs(sample)
        --print("data", torch.mean(self.input:view(self.opt.batchSize, 3, -1), 3))

        local output = self.model:forward(self.input)
        local loss = self.criterion:forward(self.model.output, self.target)

        self.model:zeroGradParameters()
        self.criterion:backward(self.model.output, self.target)
        self.model:backward(self.input, self.criterion.gradInput)

        -- update for different learning rate
        for i=1, #self.params do
            local feval = function(x)
                return self.criterion.output, self.gradParams[i]
            end
            optim.sgd(feval, self.params[i], self.optimStates[i])
        end

        local _, hitAll, hitOne, _, individual = self:computeScore(output:float(), self.target:int(), 1)
        hitAllSum = hitAllSum + hitAll
        hitOneSum = hitOneSum + hitOne
        if not individualSum then
            individualSum = individual
        else
            individualSum = torch.cat(individualSum, individual:view(1, -1), 1)
        end
        lossSum = lossSum + loss
        N = N + 1

        local trainTime = timer:time().real
        local totalTime = dataTime + trainTime
        print((' | Train [%d/%d] hitAll: %3.3f, hitOne: %3.3f, loss: %1.4f, dataTime: %.3f, trainTime: %.3f, totalTime: %.3f'):format(n, size, hitAll, hitOne, loss, dataTime, trainTime, totalTime))
        local info = ' | Individual:'
        for i=1, individual:nElement() do
            info = info .. ' ' .. ('%3.3f'):format(individual:squeeze()[i])
        end
        print(info)

        timer:reset()
    end

    local trainHitAll = hitAllSum / N
    local trainHitOne = hitOneSum / N
    local trainLoss = lossSum / N
    local trainIndividual = torch.mean(individualSum, 1):float()
    print((' | Train Done, hitAllAvg: %3.3f, hitOneAvg: %3.3f, lossAvg: %1.4f'):format(trainHitAll, trainHitOne, trainLoss))
    local info = ' `-> IndividualAvg:'
    for i=1, trainIndividual:nElement() do
        info = info .. ' ' .. ('%3.3f'):format(trainIndividual:squeeze()[i])
    end
    print(info)
end

function Trainer:val()
    local size = self.valDataLoader:size()
    local nCrops = self.opt.tenCrop and 10 or 1

    local hitEachSum, hitAllSum, hitOneSum, top1Sum, lossSum = 0.0, 0.0, 0.0, 0.0, 0.0
    local individualSum = nil

    local N = 0
    self.model:evaluate()
    local timer = torch.Timer()
    cost_time = {}
    for n, sample in self.valDataLoader:nextBatch() do
        self:copyInputs(sample)

        timer:reset()
        local output = self.model:forward(self.input)
        local testTime = timer:time().real
        table.insert(cost_time, testTime)
        local loss = self.criterion:forward(self.model.output, self.target)

        local hitEach, hitAll, hitOne, top1, individual = self:computeScore(output:float(), self.target:int(), nCrops)

        hitEachSum = hitEachSum + hitEach
        hitAllSum = hitAllSum + hitAll
        hitOneSum = hitOneSum + hitOne
        top1Sum = top1Sum + top1
        lossSum = lossSum + loss
        if not individualSum then
            individualSum = individual
        else
            individualSum = torch.cat(individualSum, individual:view(1, -1), 1)
        end

        N = N + 1
        print((' | Val [%d/%d] hitEach: %3.3f, hitAll: %3.3f, hitOne: %3.3f, top1: %3.3f, loss: %1.4f, fps: %3.3f')
            :format(n, size, hitEach, hitAll, hitOne, top1, loss, self.opt.batchSize / testTime))
    end

    local valHitEach = hitEachSum / N
    local valHitAll = hitAllSum / N
    local valHitOne = hitOneSum / N
    local valTop1 = top1Sum / N
    local valLoss = lossSum / N
    local valIndividual = torch.mean(individualSum, 1):float()
    local fps = self.opt.batchSize / torch.mean(torch.Tensor(cost_time))

    print((' * Val Done, hitEachAvg: %3.3f, hitAllAvg: %3.3f, hitOneAvg: %3.3f, top1Avg: %3.3f, lossAvg: %1.4f, fpsAvg: %3.1f')
        :format(valHitEach, valHitAll, valHitOne, valTop1, valLoss, fps))
    local info = ' `-> IndividualAvg:'
    for i=1, valIndividual:nElement() do
        info = info .. ' ' .. ('%3.3f'):format(valIndividual:squeeze()[i])
    end
    print(info)

    vals = {valHitEach, valHitAll, valHitOne, valTop1, valLoss}
    return vals
end

function Trainer:computeScore(output, target, nCrops)
    local output = output:view(output:size(1) / nCrops, nCrops, output:size(2)):sum(2):squeeze(2) / nCrops
    local batchSize = output:size(1)

    local predict = torch.ge(output, self.opt.threadshold):int()
    local positive = torch.eq(target, 1):int()

    local hitEach = torch.mean(torch.eq(predict, target):float())
    local hitAll = torch.mean(torch.eq(torch.sum(torch.eq(predict, target):int(), 2), self.opt.nClasses):float())
    local hitOne = torch.mean(torch.ge(torch.sum(torch.cmul(predict, positive), 2), 1):float()) 
    local individual = torch.mean(torch.eq(predict, target):float(), 1)

    local _, sortedPrediction = output:float():sort(2, true)
    local sortedTarget = target:gather(2, sortedPrediction)   
    local top1 = torch.mean(sortedTarget:narrow(2, 1, 1):float())

    return hitEach*100, hitAll*100, hitOne*100, top1*100, individual*100
end

function Trainer:copyInputs(sample)
    if self.opt.device == 'gpu' then
        self.input = sample.input:cuda()
        self.target = sample.target:cuda()
    else
        self.input = sample.input
        self.target = sample.target
    end
end

function Trainer:learningRate(epoch)
    local decay = epoch>= 40 and 3 or epoch >= 30 and 2 or epoch >=15 and 1 or 0
    return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
