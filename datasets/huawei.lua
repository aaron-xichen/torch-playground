local t = require 'datasets/transforms'

local M = {}
local huaweiDataset = torch.class('sl.huaweiDataset', M)

function huaweiDataset:__init(imageInfo, opt, split)
    assert(imageInfo[split], split)
    self.imageInfo = imageInfo[split]
    self.split = split
    self.meanstd = imageInfo['meanstd']
    self.opt = opt
end

function huaweiDataset:get(i)
    local image = self.imageInfo.data[i]:float()
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
        if self.opt.externelMean then
            return t.Compose{
                t.RandomCrop(224, 0),
                t.SubstractMean(self.opt.externelMean),
                t.HorizontalFlip(0.5)
            }
        else
            return t.Compose{
                t.RandomCrop(224, 0),
                t.ColorNormalize(self.meanstd),
                t.HorizontalFlip(0.5)
            }
        end
    elseif self.split == 'val' then
        if self.opt.externelMean then
            return t.Compose{
                t.RandomCrop(224, 0),
                t.SubstractMean(self.opt.externelMean)
            }
        else
            return t.Compose{
                t.RandomCrop(224, 0),
                t.ColorNormalize(self.meanstd)
            }
        end
    else
        error('invalid split: ' .. self.split)
    end
end

return M.huaweiDataset
