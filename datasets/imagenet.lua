local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'
local utee = require 'utee'

local M = {}
local ImagenetDataset = torch.class('nn.ImagenetDataset', M)

function ImagenetDataset:__init(imageInfo, opt, split)
    self.imageInfo = imageInfo[split]
    self.opt = opt
    self.split = split
    self.dir = paths.concat(opt.data, split)

    if paths.filep(self.opt.meanfilePath) then
        print("Loading from externel mean file " .. self.opt.meanfilePath)
        self.meanstd = torch.load(self.opt.meanfilePath)
    else
        print("Using internal default mean file")
        self.meanstd = {
            factor = 255.0,
            mean = {0, 0, 0},
            std = {1.0, 1.0, 1.0}
        }
    end
    print(self.meanstd)
    assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function ImagenetDataset:get(i)
    local path = ffi.string(self.imageInfo.imagePath[i]:data())

    local image = self:_loadImage(paths.concat(self.dir, path))
    local class = self.imageInfo.imageClass[i]

    return {
        input = image,
        target = class,
    }
end

function ImagenetDataset:_loadImage(path)
    local ok, input = pcall(function()
            local val = image.load(path, 3, 'float'):mul(self.meanstd.factor)
            return val
        end)

    -- Sometimes image.load fails because the file extension does not match the
    -- image format. In that case, use image.decompress on a ByteTensor.
    if not ok then
        print('!!!!!')
        local f = io.open(path, 'r')
        assert(f, 'Error reading: ' .. tostring(path))
        local data = f:read('*a')
        f:close()

        local b = torch.ByteTensor(string.len(data))
        ffi.copy(b:data(), data, b:size(1))

        input = image.decompress(b, 3, 'float')
    end

    return input
end

function ImagenetDataset:size()
    return self.imageInfo.imageClass:size(1)
end

local pca = {
    eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
    eigvec = torch.Tensor{
        { -0.5675,  0.7192,  0.4009 },
        { -0.5808, -0.0045, -0.8140 },
        { -0.5836, -0.6948,  0.4203 },
    },
}

function ImagenetDataset:preprocess()
    if self.split == 'train' then
        return t.Compose{
            t.RandomSizedCrop(224),
            t.ColorJitter({
                    brightness = 0.4,
                    contrast = 0.4,
                    saturation = 0.4,
                }),
            t.Lighting(0.1, pca.eigval, pca.eigvec),
            t.ColorNormalize(self.meanstd),
            t.HorizontalFlip(0.5),
        }
    elseif self.split == 'val' then
        local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
        return t.Compose{
            t.Scale(256),
            t.ColorNormalize(self.meanstd),
            Crop(224),
            t.Cast(self.meanstd.factor), -- whether to cast to int-float
        }
    else
        error('invalid split: ' .. self.split)
    end
end

return M.ImagenetDataset
