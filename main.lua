require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
local utee = require 'utee'

torch.setdefaulttensortype('torch.FloatTensor')

-- generate shareOpts and projectOpts objects
local shareOpts = require 'shareOpts'
local projectName = shareOpts.getProject(arg)
local projectOpts = require('project/' .. projectName .. '/opts')

-- parse the args
local cmd = shareOpts.option()
cmd = projectOpts.option(cmd)
local opt = shareOpts.parse(cmd, arg)
opt = projectOpts.parse(opt)

-- set seed
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Loading data first, because we need to infer the nClasses from the dataset
local DataLoader = require('project/' .. opt.project .. '/' .. opt.dataloader)
local trainLoader, valLoader = DataLoader.create(opt)


-- Creating model
local models = require('project/' .. opt.project .. '/models/init')
local model, criterion, optimState, epoch = models.setup(opt)

local Trainer = require('project/' .. opt.project .. '/train')
local trainer = Trainer(model, criterion, optimState, opt)

-- get monitor keys and init val
local bestKeys, bestVals = trainer:getBestStat()
local higherIsBetter = 2 * torch.lt(torch.Tensor(bestVals), 0):float() - 1

local startEpoch = epoch or 1
for epoch = startEpoch, opt.nEpochs do
    if not opt.testOnly then
        trainer:train(epoch, trainLoader)
    end
    local vals = trainer:val(valLoader)
    assert(#vals == #bestVals, 'Size is not match')
    
    local findNewBest = false
    for i=1, #vals do
        if (vals[i] - bestVals[i]) * higherIsBetter[i] > 0 then
            bestVals[i] = vals[i]
            findNewBest = true
        end
    end

    if opt.testOnly then
        break
    end

    if findNewBest then
        print('... New best model, ' .. utee.convertToString(bestKeys, bestVals))
        --save(model, trainer.optimState, epoch, opt.workspace .. '/best.t7')
    end
end

print('=> Finish! Best model, ' .. utee.convertToString(bestKeys, bestVals))