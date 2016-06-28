local M = { }

function M.option(cmd)
    cmd:text('Torch-7 Huawei Scene Multi-label Classification Arguments Options:')
    cmd:option('-lrRatio',              0.01,      'The ratio between pretrained layers and new layer')
    cmd:option('-lossWeights',        'none',       'Loss weights')
    cmd:option('-threadshold',         0.5,       'Threadshold')
    cmd:option('-modelRoot', 'none', 'Model root, must contains deploy.prototxt and weights.caffemodel')
    --cmd:option('-loadType',     'cudnn',              'Options: nn | cudnn')
    cmd:option('-trainListPath', '/home/chenxi/dataset/huawei_scene_labeling/train.labels', 'Train list path')
    cmd:option('-valListPath', '/home/chenxi/dataset/huawei_scene_labeling/val.labels', 'Val list path')
    cmd:option('-imgRoot', '/home/chenxi/dataset/huawei_scene_labeling/resized256/', 'Image root path')
    cmd:option('-externelMean',        'none',  'Externel mean file path')
    cmd:text()
    return cmd
end

function M.parse(cmd, opt)
    opt.lossWeights = torch.Tensor{2187, 1727, 152, 679, 167, 17, 112, 60, 633, 397, 61, 455, 69, 8}
    opt.lossWeights = opt.lossWeights / torch.sum(opt.lossWeights) * opt.lossWeights:nElement()

    local info = 'Loss weights:'
    for i=1, opt.lossWeights:nElement() do
        info = info .. ' ' .. ('%3.3f'):format(opt.lossWeights:squeeze()[i])
    end
    print(info)

    assert(opt.modelRoot ~= 'none')
    if opt.externelMean ~= 'none' then
        print('Loading externel meanfile from ' .. opt.externelMean)
        opt.externelMean = torch.load(opt.externelMean)
    else
        print('Using internel meanfile')
    end

    if opt.loadType == 'cudnn' then
        print("Running on GPU")
    else
        print("Running on CPU")
    end

    if not opt.testOnly then
        print(("Finetuning from %s with LR: %.3f and lrRatio: %.6f"):format(paths.basename(opt.modelRoot), opt.LR, opt.lrRatio))
    end
    
    if opt.dataset == 'none' then
        cmd:error('Dataset required')
    end
    
    return opt
end

return M