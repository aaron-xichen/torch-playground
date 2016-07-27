local t = require 'datasets/transforms'

local M = {}
local MnistDataset = torch.class('nn.MnistDataset', M)

function MnistDataset:__init(imageInfo, opt, split)
    assert(imageInfo[split], split)
    self.imageInfo = imageInfo[split]
    self.split = split
end

function MnistDataset:get(i)
    local image = self.imageInfo.data[i]:float():div(255)
    local label = self.imageInfo.labels[i]

    return {
        input = image,
        target = label,
    }
end

function MnistDataset:size()
    return self.imageInfo.data:size(1)
end

function MnistDataset:preprocess()
    if self.split == 'train' or self.split == 'val' then
        return t.Compose{
            t.CenterCrop(28),
        }
    else
        error('invalid split: ' .. self.split)
    end
end
return M.MnistDataset
