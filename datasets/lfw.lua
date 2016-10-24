local t = require 'datasets/transforms'

local M = {}
local lfwDataset = torch.class('platform.lfwDataset', M)

function lfwDataset:__init(imageInfo, opt, split)
    assert(imageInfo[split], split)
    self.imageInfo = imageInfo[split]
    self.split = split
    self.opt = opt
    if paths.filep(self.opt.meanfilePath) then
        print("Loading from externel mean file " .. self.opt.meanfilePath)
        self.meanstd = torch.load(self.opt.meanfilePath)
    else
        print("Using internal default mean file")
        self.meanstd = {
            factor = 1.0,
            mean = {0, 0, 0},
            std = {1.0, 1.0, 1.0}
        }
    end
    print(self.meanstd)
end

function lfwDataset:get(i)
    local image = self.imageInfo.data[i]:float():mul(self.meanstd.factor)
    local label = self.imageInfo.labels[i]
    return {
        input = image,
        target = label,
    }
end

function lfwDataset:size()
    return self.imageInfo.data:size(1)
end

function lfwDataset:nClasses()
    return self.imageInfo.labels:size()[2]
end

function lfwDataset:preprocess()
    assert(self.split == 'val', 'Only support test mode')
    local cropType = self.opt.crop == 10 and t.TenCrop or t.CenterCrop
    return t.Compose{
        t.ColorNormalize(self.meanstd),
        cropType(224),
    }
end

return M.lfwDataset
