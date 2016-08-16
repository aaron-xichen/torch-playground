local M = { }

function M.option(cmd)
    cmd:text('Torch-7 Fixed Point Arguments Options:')
    cmd:option('-modelRoot',       'none',    'Externel model folder')
    cmd:option('-shiftInfoPath',   'shiftInfo.t7',     'Load path of shift bits, including weights, bias and activation')
    cmd:option('-metaInfoPath',    'meta.t7',          'Save path of all info')
    cmd:text()
    return cmd
end

function M.parse(cmd, opt)
    if opt.modelRoot == 'none' then
        cmd:error('model root required')
    end
    if opt.accBitWidth ~= -1 then
        assert(opt.shiftInfoPath ~= 'none', 'Please specify shiftInfoPath')
        print(("Reading shift table from %s"):format(opt.shiftInfoPath))
        opt.shiftTable = torch.load(opt.shiftInfoPath)
    end

    opt.netPath = opt.modelRoot .. '/deploy.prototxt'
    opt.modelPath = opt.modelRoot .. '/weights.caffemodel'
    opt.meanfilePath = opt.modelRoot .. '/meanfile.t7'
    opt.torchModelPath = opt.modelRoot .. '/modelCPU.t7' -- support cpu model only

    return opt
end

return M