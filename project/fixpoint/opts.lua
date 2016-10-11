local utee = require 'utee'
local M = { }

function M.option(cmd)
    cmd:text('Torch-7 Fixed Point Arguments Options:')
    cmd:option('-modelRoot',       'none',    'Externel model folder')
    cmd:option('-bitWidthConfigPath', 'none',  'Setting file path of bitwidth')
    cmd:option('-metaTablePath', 'meta.config', 'weightShift, biasShift, actShift, biasAlign, winShift, decPosSave, decPosRaw')
    cmd:text()
    return cmd
end

function M.parse(cmd, opt)
    if opt.modelRoot == 'none' then
        cmd:error('model root required')
    end
    assert(opt.metaTablePath, 'Please specify metatTablePath')
    opt.metaTable = utee.loadTxt(opt.metaTablePath)

    assert(opt.bitWidthConfigPath ~= 'none', 'Please specify bitWidthConfigPath')
    opt.bitWidthConfig = utee.loadTxt(opt.bitWidthConfigPath)
    
    opt.netPath = opt.modelRoot .. '/deploy.prototxt'
    opt.modelPath = opt.modelRoot .. '/weights.caffemodel'
    opt.meanfilePath = opt.modelRoot .. '/meanfile.t7'
    opt.torchModelPath = opt.modelRoot .. '/modelCPU.t7' -- support cpu model only

    return opt
end

return M