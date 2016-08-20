local M = { }

function M.option(cmd)
    cmd:text('Torch-7 Huawei Scene Multi-label Classification Arguments Options:')
    cmd:option('-last',                 0,        'The number of last layers to finetune')
    cmd:option('-lrRatio',              0.01,      'The ratio between pretrained layers and new layer')
    cmd:option('-lossWeights',        'none',       'Loss weights')
    cmd:option('-threadshold',         0.5,       'Threadshold')
    cmd:option('-modelRoot', 'none', 'Model root, must contains deploy.prototxt and weights.caffemodel')
    cmd:text()
    return cmd
end

function M.parse(cmd, opt)
    opt.lossWeights = torch.Tensor{2187, 1727, 152, 679, 167, 17, 112, 60, 633, 397, 61, 455, 69}
    opt.lossWeights = opt.lossWeights / torch.sum(opt.lossWeights) * opt.lossWeights:nElement()

    local info = 'Loss weights:'
    for i=1, opt.lossWeights:nElement() do
        info = info .. ' ' .. ('%3.3f'):format(opt.lossWeights:squeeze()[i])
    end
    print(info)

    assert(opt.modelRoot ~= 'none', 'Model root required')
    opt.netPath = paths.concat(opt.modelRoot, 'deploy.prototxt')
    opt.modelPath = paths.concat(opt.modelRoot, 'weights.caffemodel')
    opt.meanfilePath = paths.concat(opt.modelRoot, 'meanfile.t7')
    if opt.device == 'gpu' then
        opt.torchModelPath = paths.concat(opt.modelRoot, 'model.t7')
    else 
        opt.torchModelPath = paths.concat(opt.modelRoot, 'modelCPU.t7')
    end

    if not opt.testOnly then
        print(("Finetuning from %s with LR: %.3f and lrRatio: %.6f"):format(paths.basename(opt.modelRoot), opt.LR, opt.lrRatio))
    end

    opt.imgRoot = paths.concat(opt.data, 'resized224')
    opt.trainListPath = paths.concat(opt.data, 'train.labels')
    opt.valListPath = paths.concat(opt.data, 'val.labels')
    return opt
end

return M