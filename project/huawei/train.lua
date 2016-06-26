
local optim = require 'optim'

local M = {}
local Trainer = torch.class('huawei.Trainer', M)

function Trainer:__init(model, criterion, optimState, opt)
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
end

function Trainer:getBestStat()
    keys = {'bestHitAll', 'bestHitOne', 'bestTop1', 'bestTop3', 'bestLoss'}
    vals = {-math.huge, -math.huge, -math.huge, -math.huge, math.huge}
    return keys, vals
end

function Trainer:train(epoch, dataloader)
    for i=1, #self.params-2 do
        self.optimStates[i].learningRate = self:learningRate(epoch) * self.opt.lrRatio
    end
    self.optimStates[#self.params-1].learningRate = self:learningRate(epoch) -- for W
    self.optimStates[#self.params].learningRate = 2 * self:learningRate(epoch) -- for b
    

    local size = dataloader:size()
    local hitAllSum, hitOneSum, lossSum = 0.0, 0.0, 0.0
    local individualSum = nil

    local N = 0

    print(('=> Training epoch: #%d, LR-Former, %.3e, LR-Last: %.3e')
        :format(epoch, self:learningRate(epoch) * self.opt.lrRatio, self:learningRate(epoch)))
    self.model:training()

    local timer = torch.Timer()
    for n, sample in dataloader:nextBatch() do
        local dataTime = timer:time().real
        timer:reset()
        
        self:copyInputs(sample)
        local output = self.model:forward(self.input):float()
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
        
        local hitAll, hitOne, individual = self:computeScore(output, sample.target, 1)
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
    print((' | Train Done, hitAllAvg %3.3f, hitOneAvg: %3.3f, lossAvg: %1.4f'):format(trainHitAll, trainHitOne, trainLoss))
    local info = ' `-> IndividualAvg:'
    for i=1, trainIndividual:nElement() do
        info = info .. ' ' .. ('%3.3f'):format(trainIndividual:squeeze()[i])
    end
    print(info)
end

function Trainer:val(dataloader)
    local size = dataloader:size()
    local nCrops = self.opt.tenCrop and 10 or 1
    local hitAllSum, hitOneSum, lossSum, top1Sum, top3Sum = 0.0, 0.0, 0.0, 0.0, 0.0
    local individualSum = nil

    local N = 0
    self.model:evaluate()
    local timer = torch.Timer()
    cost_time = {}
    for n, sample in dataloader:nextBatch() do
        local input, target
        if self.opt.device == 'gpu' then
            self:copyInputs(sample)
            input = self.input
            target = self.target -- GPU automatically convert to float
        else
            input = sample.input
            target = sample.target:float() -- BCECriterion needs both float tensor
        end
        timer:reset()
        self.model:forward(input)
        local testTime = timer:time().real
        table.insert(cost_time, testTime)
        local loss = self.criterion:forward(self.model.output, target)

        local hitAll, hitOne, individual, top1, top3 = self:computeScore(self.model.output, target:int(), nCrops)

        hitAllSum = hitAllSum + hitAll
        hitOneSum = hitOneSum + hitOne
        top1Sum = top1Sum + top1
        top3Sum = top3Sum + top3
        lossSum = lossSum + loss
        if not individualSum then
            individualSum = individual
        else
            individualSum = torch.cat(individualSum, individual:view(1, -1), 1)
        end

        N = N + 1
        print((' | Val [%d/%d] hitAll: %3.3f, hitOne: %3.3f, top1: %3.3f, top3: %3.3f, loss: %1.4f, fps: %3.3f')
            :format(n, size, hitAll, hitOne, top1, top3, loss, self.opt.batchSize / testTime))
    end
    
    local valHitAll = hitAllSum / N
    local valHitOne = hitOneSum / N
    local valTop1 = top1Sum / N
    local valTop3 = top3Sum / N
    local valLoss = lossSum / N
    local valIndividual = torch.mean(individualSum, 1):float()
    local fps = self.opt.batchSize / torch.mean(torch.Tensor(cost_time))
    
    print((' * Val Done, hitAllAvg %3.3f, hitOneAvg: %3.3f, top1Avg: %3.3f, top3Avg: %3.3f, lossAvg: %1.4f, fpsAvg: %3.1f')
        :format(valHitAll, valHitOne, valTop1, valTop3, valLoss, fps))
    local info = ' `-> IndividualAvg:'
    for i=1, valIndividual:nElement() do
        info = info .. ' ' .. ('%3.3f'):format(valIndividual:squeeze()[i])
    end
    print(info)
    
    --keys = {'hitAll', 'hitOne', 'top1', 'top3', 'loss'}
    vals = {valHitAll, valHitOne, valTop1, valTop3, valLoss}
    return vals
end

function Trainer:computeScore(output, target, nCrops)
    local output = output:view(output:size(1) / nCrops, nCrops, output:size(2)):sum(2):squeeze(2) / nCrops
    local batchSize = output:size(1)

    local predict = torch.ge(output, self.opt.threadshold):int()
    local positive = torch.eq(target, 1):int()

    local hitAll = torch.mean(torch.eq(torch.sum(torch.eq(predict, target):int(), 2), self.opt.nClasses):float())
    local hitOne = torch.mean(torch.ge(torch.sum(torch.cmul(predict, positive), 2), 1):float()) 
    local individual = torch.mean(torch.eq(predict, target):float(), 1)

    local _, sortedPrediction = output:float():sort(2, true)
    local sortedTarget = target:gather(2, sortedPrediction)
    
    local top1 = torch.mean(sortedTarget:narrow(2, 1, 1):float())
    local top3 = torch.mean(torch.ge(torch.sum(sortedTarget:narrow(2, 1, 3), 2), 1):float())
    return hitAll * 100, hitOne * 100, individual * 100, top1 * 100, top3 *  100
end

function Trainer:copyInputs(sample)
    self.input = self.input or (self.opt.nGPU == 1
        and torch.CudaTensor()
        or cutorch.createCudaHostTensor())
    self.target = self.target or torch.CudaTensor()

    self.input:resize(sample.input:size()):copy(sample.input)
    self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
    local decay = epoch>= 175 and 3 or epoch >= 150 and 2 or epoch >=100 and 1 or 0
    return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
