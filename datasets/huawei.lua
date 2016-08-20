local t = require 'datasets/transforms'

local M = {}
local huaweiDataset = torch.class('platform.huaweiDataset', M)

function huaweiDataset:__init(imageInfo, opt, split)
    assert(imageInfo[split], split)
    self.imageInfo = imageInfo[split]
    self.split = split
    self.opt = opt
    
    
    print("Loading from externel mean file " .. self.opt.meanfilePath)
    self.meanstd = torch.load(self.opt.meanfilePath)
end

function huaweiDataset:get(i)
    local image = self.imageInfo.data[i]:float():mul(self.meanstd.factor)
    local label = self.imageInfo.labels[i]
    return {
        input = image,
        target = label,
    }
end

function huaweiDataset:size()
    return self.imageInfo.data:size(1)
end

function huaweiDataset:nClasses()
    return self.imageInfo.labels:size()[2]
end

function huaweiDataset:preprocess()
    if self.split == 'train' then
        return t.Compose{
            t.RandomCrop(224, 8),
            t.ColorNormalize(self.meanstd),
            t.HorizontalFlip(0.5),
        }
    elseif self.split == 'val' then
        return t.Compose{
            t.ColorNormalize(self.meanstd),
            --t.CenterCrop(224),
        }
    else
        error('invalid split: ' .. self.split)
    end
end

return M.huaweiDataset
