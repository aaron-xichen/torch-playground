local utee = require 'utee'
local M = { }

function M.option(cmd)
    cmd:text('Torch-7 Quantization Arguments Options:')
    cmd:option('-modelRoot',       'none',    'Externel model folder')
    cmd:option('-collectNSamples',  10,      'Number of samples to collect')
    cmd:option('-bitWidthConfigPath', 'none',  'Setting file path of bitwidth')
    cmd:option('-metaTablePath', 'meta.config', 'weightShift, biasShift, actShift, biasAlign, winShift, decPosSave, decPosRaw')
    cmd:text()
    return cmd
end

function M.parse(cmd, opt)
    -- require model path
    if opt.modelRoot == 'none' then
        cmd:error('model root required')
    end
    opt.netPath = opt.modelRoot .. '/deploy.prototxt'
    opt.modelPath = opt.modelRoot .. '/weights.caffemodel'
    opt.meanfilePath = opt.modelRoot .. '/meanfile.t7'
    if opt.device == 'gpu' then
        opt.torchModelPath = opt.modelRoot .. '/model.t7'
    else 
        opt.torchModelPath = opt.modelRoot .. '/modelCPU.t7'
    end
    
    -- require bit-width configuration
    if opt.bitWidthConfigPath ~= 'none' then
        opt.bitWidthConfig = utee.loadTxt(opt.bitWidthConfigPath)
    end
    if paths.filep(opt.metaTablePath) then
        opt.metaTable = utee.loadTxt(opt.metaTablePath)
        opt.metaTableExist = true
    end
    return opt
end

return M
